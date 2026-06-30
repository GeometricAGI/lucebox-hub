# DFlash feature ring — ideas, findings & next steps

> Handoff note for future sessions. Captures what `server/src/common/dflash_feature_ring.cpp`
> does, the optimizations attempted on branch `pramodith/ring_transfer`, what was measured
> (and why most of it was flat on single-GPU), and where the realistic gains actually are.
>
> Branch at time of writing: `pramodith/ring_transfer` (base commit `1b11c50`).
> Test hardware: single **RTX 3090** (sm_86, 24 GB). BSA spec-prefill auto-disabled on sm_86.

---

## TL;DR

- `dflash_feature_ring.cpp` is a **data-transport + storage-format layer** between the target's
  per-layer hidden states ("features") and the DFlash draft that consumes them. It is **not** a
  compute hotspot.
- On **single GPU**, feature movement is <1% of a decode step (the step is compute-bound on the
  target verify forward). **No optimization in this file moves single-GPU throughput** — confirmed
  by three independent A/Bs, all flat within noise.
- Its leverage is real but **conditional on a device/process boundary** (multi-GPU / layer-split),
  where the feature handoff becomes a per-step PCIe transfer. That is the regime to measure in.
- Secondary lever: **VRAM** via `DFLASH_FEATURE_DTYPE` (storage dtype), with an AL/throughput tradeoff.
- For single-GPU *speed*, look at `verify_compute` (77% of the step), acceptance length (draft
  quality), and the logits/argmax path — not this file.

---

## What the feature ring is

The DFlash draft predicts tokens from the target model's **hidden states** (concatenation of a few
captured target layers per token, `fc_in = n_capture_layers × hidden`). `DraftFeatureMirror` is a
ring buffer (`target_feat`, a 2D tensor `[fc_in, cap]`) holding those features per committed token.

Measured shape on Qwen3.6-27B (server, default config):
`mirror dtype=f32 cap=4096 fc_in=25600` → 25,600 = 5 capture layers × 5,120 hidden → ~100 KB/token,
ring ≈ 420 MB at F32.

**Why "needing the data" ≠ "needing to copy it":** on one GPU, producer (target) and consumer
(draft) share the device, so the draft reads features in place via the **mirror-view**
(`draft_feature_mirror_can_view`, requires F32 + contiguous window → builds a ggml view into
`target_feat`, zero copy). The ring/copy machinery only becomes a real transfer when a device or
process boundary separates producer and consumer.

---

## CRITICAL: two different feature paths (this tripped up measurement)

There are **two writers** and the binary you run decides which is exercised:

| Path | Binary | Writer | Reader |
|---|---|---|---|
| **Layer-split forward** | `test_dflash` (and multi-GPU shards) | `copy_capture_slice_to_draft_ring` (captures activations during the forward) | `copy_feature_ring_range_to_tensor` (in-proc), `copy_feature_ring_range_to_host_f32` (remote/IPC draft) |
| **In-process backend** | `dflash_server` | `draft_feature_mirror_sync_range` → `convert_bf16_feature_to_storage` (mirrors from the target's `cache_.target_feat` after compute) | same readers |

- `test_dflash` (what `bench_llm.py` and `speed_profile.py` wrap) runs `run_qwen35_layer_split_forward`
  → uses `copy_capture_slice` + `copy_feature_ring_range_to_tensor`. Its config prints
  `draft_feature_mirror=0` and the per-step timer shows **`mirror_sync 0.00`** — i.e. it does **not**
  call `convert_bf16_feature_to_storage`.
- `dflash_server` (driven by `bench_daemon.py`) uses `qwen35_backend` → `sync_range` →
  `convert_bf16_feature_to_storage`. Callers of `sync_range`/`sync_tail` are the in-process backends
  only: `qwen35_backend`, `gemma4_backend`, `laguna_backend`, `qwen35moe_backend`.

**Lesson:** match the harness to the function. A change to `convert_bf16_feature_to_storage` is
invisible to `bench_llm.py` because `test_dflash` never calls it.

---

## Ideas attempted (all on branch `pramodith/ring_transfer`)

All three are **correct and lossless** (verified or numerically exact) but **perf-neutral on single GPU**.

1. **Async copy-stream overlap for the writer** (`copy_capture_slice_to_draft_ring`, + `copy_stream`
   / `compute_done_event` in the struct). Capture layer L's hidden into the ring on a dedicated
   non-blocking stream while layer L+1 computes on the default stream; readers
   (`copy_feature_ring_range_to_tensor` / `_to_host_f32`) `cudaStreamSynchronize(copy_stream)` before
   reading. Removes a per-capture `cudaDeviceSynchronize`. **Payoff: layer-split only** (needs the
   per-layer pipeline seam; the single-GPU forward is one fused ggml graph with no seam).

2. **Bulk ≤2-segment reader** (`copy_feature_ring_range_to_host_f32`). A ring range wraps at most
   once → ≤2 contiguous segments. Replaced one `ggml_backend_tensor_get` *per token* (≈ n_tokens D2H
   calls for a 2048-token window) with ≤2 bulk D2H transfers pipelined on the copy stream; F32 copies
   straight into `out`, other dtypes stage raw then host-convert. **Payoff: remote/cross-device draft
   path** (`qwen35_layer_split_adapter::snapshot_draft_features`); not on single-GPU's path.
   Caveat: `out` is pageable `std::vector`, so transfers don't truly overlap compute without pinning.

3. **On-device bf16→f32 convert** (`convert_bf16_feature_to_storage`, default F32-storage branch).
   Replaced the host round-trip (bf16 D2H → CPU widen → f32 H2D, chunked) with one on-GPU upconvert
   via `ggml_get_to_fp32_cuda(GGML_TYPE_BF16)`, mirroring what the reader (`copy_feature_to_f32`)
   already does. bf16→f32 is an exact widen → bit-identical. Host path kept for f16/q8_0 storage and
   if the CUDA converter is unavailable. **Exercised on `dflash_server`** (validated: ran correctly,
   coherent output). Payoff lands on multi-GPU where that round-trip crosses PCIe.

---

## Measurements (single RTX 3090)

**Per-step phase breakdown** (`test_dflash`, 47 steps, the smoking gun):
```
verify_compute  47.82 ms  (77%)   ← target forward over the draft tokens
draft_compute    6.48 ms  (10%)
verify_logits    5.32 ms   (9%)   ← projection + argmax over 248K vocab
verify_build     1.53 ms
draft_copyfeat   0.57 ms  (0.9%)  ← entire reader side of this file
mirror_sync      0.00 ms          ← entire convert_bf16 writer side
-----   sum     62.26 ms
```

**A/B results (all within run-to-run noise):**

| Harness | Metric | Baseline | With change | Δ |
|---|---|---|---|---|
| `speed_profile.py` (test_dflash, ring patch) | decode tok/s | 83.61 ± 1.47 | 83.25 ± 1.48 | −0.4% |
| `bench_llm.py` HumanEval (test_dflash, bf16) | DFlash tok/s | 96.40 | 96.42 | +0.02 |
| `bench_llm.py` HumanEval (test_dflash, bf16) | AL | 6.20 | 6.20 | 0 (lossless) |
| `dflash_server` + `bench_daemon.py` (bf16) | decode tok/s | 35.19 | 35.34 | +0.4% |

- `speed_profile` nsys A/B for the ring patch: **identical** kernel/memcpy/sync call counts → the
  changed path wasn't even exercised single-GPU.
- bf16 A/B per-prompt AL identical on all 10 prompts → lossless confirmed.
- The lossless gate "failures" on `rolling_max`/`sum_product` are **pre-existing** batched-verify FP
  nondeterminism (baseline fails the same prompts), not caused by any change.
- `dflash_server` decode (~35 tok/s) is much lower than `test_dflash` (~96) because the server runs
  TQ3_0 KV-quant + chat template + full HTTP path; only compare like-for-like within a harness.

**Why flat:** Amdahl. The optimizations shrink data-movement/serialization cost; on single GPU that
cost is ~0–0.6 ms of a ~62 ms compute-bound step. Large numerator only exists across a GPU boundary.

---

## Where the realistic gains are

1. **Multi-GPU / layer-split (this file's reason to exist).** Target shards on different GPUs/
   processes than the draft → feature handoff is a real per-step PCIe transfer (~100 KB/token ×
   draft_ctx). The three changes above target exactly this. **Needs a ≥2-GPU testbed to measure.**
2. **VRAM (not speed).** `DFLASH_FEATURE_DTYPE=f16|bf16|q8_0` shrinks the ~420 MB ring (½, ½, ¼).
   Frees VRAM for KV cache / larger `cap`. Tradeoff: non-F32 **disables the mirror-view** (forces a
   read-time copy/convert) and may lower AL.
3. **Single-GPU speed lives elsewhere** (not this file):
   - `verify_compute` (77%): attention/GEMM kernels, TQ3_0 KV dequant cost; **BSA spec-prefill is
     disabled on sm_86** — active on newer GPUs.
   - **Acceptance length** (highest leverage): higher AL = fewer verify steps/token. Function of
     *draft quality*, not this file. AL 6.2 → 8 beats anything achievable here.
   - `verify_logits` (9%): GPU top-k/argmax path (`DFLASH27B_*` env flags) to avoid logits D2H.

---

## Multi-GPU next steps (the move the user is making)

- Launch the layer-split / target-shard daemons: `backend_ipc_main` with
  `--backend-ipc-mode=qwen35-target-shard <target.gguf> --target-gpu=N --layer-begin=N --layer-end=N`
  and `--backend-ipc-mode=dflash-draft <draft> --draft-gpu=M`. (`server/src/ipc/backend_ipc_main.cpp`,
  `--draft-ipc-bin` on `dflash_server` for mixed backends.)
- Put target shard on GPU 0, draft on GPU 1 so the feature handoff actually crosses the boundary.
- Drive with `dflash_server` + `bench_daemon.py`; the **proof the change is live** is the nsys
  `Memcpy DtoH/HtoD` count and time per token dropping, plus `mirror_sync`/feature phases shrinking.
- **Worth doing for real overlap on this path:** pin the host staging buffer in
  `copy_feature_ring_range_to_host_f32` (`snap.data` is a pageable `std::vector` →
  `cudaHostRegister`/pinned alloc) so the ≤2-segment D2H transfers actually overlap compute.
- Consider making the draft compute wait on `copy_stream` via `cudaStreamWaitEvent` (a 1-line backend
  edit before the mirror-view draft `graph_compute`) so writes can be truly fire-and-forget instead
  of synced at `sync_range` end — needed to get the full write/compute overlap.

---

## How to reproduce the A/B (reusable recipe)

```bash
# isolate a single change: snapshot the file, build baseline, build with-change, compare
git diff HEAD -- server/src/common/dflash_feature_ring.cpp > /tmp/change.patch
git checkout HEAD -- server/src/common/dflash_feature_ring.cpp   # or restore a no-change version
cmake -B server/build -S server -DCMAKE_BUILD_TYPE=Release
cmake --build server/build --target test_dflash test_generate dflash_server -j

# test_dflash harness (NOTE: does NOT exercise convert_bf16_feature_to_storage)
.venv/bin/python server/scripts/bench_llm.py --bench HumanEval

# server harness (exercises the in-process backend / convert_bf16 path)
DFLASH27B_KV_TQ3=1 ./server/build/dflash_server server/models/Qwen3.6-27B-Q4_K_M.gguf \
  --draft server/models/draft/dflash-draft-3.6-q4_k_m.gguf --ddtree --ddtree-budget 22 --port 8000 &
.venv/bin/python server/scripts/bench_daemon.py --url http://localhost:8000 --n-gen 256 --warmup
```

Gotchas learned:
- Python env is the project venv at `/workspace/lucebox-hub/.venv` (run via `uv run` or call its
  python directly); `transformers`/`datasets` live there, not in `/venv/main`.
- `pkill -f dflash_server` **self-matches the shell** running it (exit 144) — kill by PID instead.
- A fresh git worktree needs submodules (`git submodule update --init server/deps/llama.cpp
  server/deps/Block-Sparse-Attention`) and won't have the gitignored 16 GB models — point
  `DFLASH_TARGET`/`DFLASH_DRAFT` at the main tree.
