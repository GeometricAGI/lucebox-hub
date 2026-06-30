# Lucebox speed profile

- created: `2026-06-29T13:50:21.023334+00:00`
- commit: `1b11c50`
- gpu: `NVIDIA GeForce RTX 3090`  driver: `595.58.03`  power: `350.00 W`
- target: `Qwen3.6-27B-Q4_K_M.gguf`  draft: `dflash-draft-3.6-q4_k_m.gguf`
- n_gen: `128`  budget: `22`  reps: `5`  nsys: `True`

## Headline

| metric | value |
|---|---:|
| AR decode (tok/s) | 33.94 |
| DFlash decode (tok/s) | 86.04 ± 0.51 |
| speedup vs AR | 2.54x |
| ms / token | 12.18 ± 0.05 |
| TTFT estimate (ms) | 1060.5 ± 17.8 |
| prefill (ms) | 976.7 |
| acceptance length (AL) | 5.37 ± 0.00 |
| accept % / step | 33.5 |
| decode tok/s spread (min–max) | 62.2–101.9 |

## Noise / detection threshold

- ✅ **stable** — all tracked relative stddevs are at or below `5.0%`.

## Repeated-run samples by prompt

| prompt | decode tok/s | ms/token | AL | TTFT ms |
|---|---:|---:|---:|---:|
| `has_close_elements` | 93.96 ± 0.49 | 10.64 ± 0.06 | 5.82 ± 0.00 | 1051.3 ± 18.8 |
| `rolling_max` | 101.94 ± 0.74 | 9.81 ± 0.07 | 6.40 ± 0.00 | 1058.2 ± 17.1 |
| `sum_product` | 62.22 ± 0.09 | 16.07 ± 0.02 | 3.88 ± 0.00 | 1071.9 ± 17.3 |

## Correctness (losslessness gate)

- **FAIL — spec-decode output differs from greedy AR on 2/3 prompts (rolling_max, sum_product), first at token #27. AR is self-deterministic, so this is NOT run-to-run noise — but not proven a logic bug either: it can be batched-verify FP (verify scores draft tokens as a batch vs AR one-at-a-time). Classify via the logit gap at the divergence: near-tie = FP, clear gap = bug.**
- FAIL = output changed and it is not run-to-run noise (AR agreed with itself); inconclusive prompts (AR non-deterministic) are excluded. FP-vs-bug needs the logit gap at the first mismatch (a follow-up the binaries don't emit yet).

## Per-step phase breakdown (engine timers, ms/step)

| phase | ms/step | % of step |
|---|---:|---:|
| `verify_compute` | 48.01 | 78% |
| `draft_compute` | 6.63 | 11% |
| `verify_logits` | 4.33 | 7% |
| `verify_build` | 1.55 | 3% |
| `draft_copyfeat` | 0.55 | 1% |
| `draft_build` | 0.32 | 1% |
| `draft_logits` | 0.16 | 0% |
| `verify_set` | 0.14 | 0% |
| `draft_set` | 0.04 | 0% |
| `replay_set` | 0.00 | 0% |
| `draft_bridge` | 0.00 | 0% |
| `snapshot_ssm` | 0.00 | 0% |
| `replay_logits` | 0.00 | 0% |
| `restore_ssm` | 0.00 | 0% |
| `mirror_sync` | 0.00 | 0% |
| `accept` | 0.00 | 0% |
| `replay_build` | 0.00 | 0% |
| `replay_compute` | 0.00 | 0% |
| **sum** | **61.74** | 100% |

## Kernel-level (nsys)

- GPU kernel time: `1240.87 ms`  over `128` tokens
- kernel launches/token: `531.8`  (fusion signal)
- host<->device copy: `1585.25 ms` total, `12.385 ms/token` (overlap signal)

**Top kernels by GPU time**

| kernel | total ms | launches | avg µs |
|---|---:|---:|---:|
| `void mul_mat_q<(ggml_type)12, (int)24, (bool)0>(const char *, const in` | 459.96 | 6336 | 72.59 |
| `void mul_mat_q<(ggml_type)14, (int)24, (bool)0>(const char *, const in` | 184.87 | 1430 | 129.28 |
| `void gated_delta_net_cuda<(int)128, (bool)0, (bool)1, __half>(const fl` | 83.88 | 1056 | 79.43 |
| `void mul_mat_q<(ggml_type)13, (int)24, (bool)0>(const char *, const in` | 46.97 | 1056 | 44.48 |
| `void mul_mat_q<(ggml_type)14, (int)16, (bool)0>(const char *, const in` | 43.96 | 110 | 399.62 |
| `void mul_mat_q<(ggml_type)12, (int)80, (bool)0>(const char *, const in` | 40.52 | 336 | 120.59 |
| `void mul_mat_q<(ggml_type)12, (int)16, (bool)0>(const char *, const in` | 37.85 | 682 | 55.5 |
| `void mul_mat_q_stream_k_fixup<(ggml_type)12, (int)24, (bool)0>(const i` | 35.61 | 6336 | 5.62 |

**Sync / launch / copy CUDA APIs** (CPU-stall signal)

| api | total ms | calls |
|---|---:|---:|
| `cudaMemcpyAsync` | 1631.47 | 7791 |
| `cudaStreamSynchronize` | 662.33 | 3898 |
| `cudaLaunchKernel` | 608.65 | 63649 |
| `cudaLaunchKernelExC_v11060` | 13.28 | 2208 |
| `cudaMemcpy2DAsync` | 8.66 | 1920 |
| `cudaDeviceSynchronize` | 2.23 | 31 |
| `cudaMemcpy` | 0.46 | 44 |

## Where the margin is

- `verify_compute` is 78% of the step (48.0 ms) — primary target.
- 531.8 kernel launches/token — kernel-fusion candidate (launch overhead dominates many tiny kernels).
- 12.38 ms/token in host<->device copies — CPU/GPU-overlap candidate (data shuttling off the critical path).
