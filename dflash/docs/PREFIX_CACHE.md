# Prefix Cache Design

## Overview

The prefix cache accelerates multi-turn conversations by snapshotting the
KV cache state at turn boundaries. On subsequent requests that share the
same system prompt / early conversation, the server restores from a snapshot
instead of recomputing the full prefill — saving both latency and compute.

```
Request 1: [system + user1 + assistant1 + user2]
                    ↑ boundary — snapshot here
Request 2: [system + user1 + assistant1 + user2 + assistant2 + user3]
            └── restore snapshot ──────────┘      └── diff-prefill ──┘
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    HTTP Server                           │
│  • Tokenizes prompt                                     │
│  • Calls PrefixCache for lookup / prepare               │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              PrefixCache  (LRU logic)                    │
│  • detect chat boundaries via ChatMarkers               │
│  • SHA-1 hash prefix at each boundary                   │
│  • LRU eviction when cap is reached                     │
│  • Two tiers: inline prefix + full-compress             │
└────────────────────────┬────────────────────────────────┘
                         │ snapshot_save / restore_and_generate
                         ▼
┌─────────────────────────────────────────────────────────┐
│             ModelBackend (per-arch)                      │
│  • snapshot_save(slot): copy KV cache → snap_backend_   │
│  • restore_and_generate(slot, req): snap → KV + decode  │
│  • snapshot_free(slot): release snapshot memory          │
└─────────────────────────────────────────────────────────┘
```

## Two-Tier Caching

### Tier 1: Inline Prefix Cache

Caches KV state at **turn boundaries** within a conversation. The boundary
detector uses `ChatMarkers` to find end-of-message + start-of-next-role
token sequences.

- **lookup()**: Finds the longest cached prefix matching the current prompt.
- **prepare_inline_snap()**: Selects a slot and cut-point for snapshotting
  after the current prefill completes.
- **confirm_inline_snap()**: Commits the entry after successful save.
- LRU eviction: oldest entry's slot is reused when capacity is reached.

### Tier 2: Full-Compress Cache

Caches the **entire post-compression KV state**, keyed on the raw
(pre-compression) prompt. Hits skip both PFlash compression and prefill
entirely — the fastest path.

- Separate slot pool starting at `cap` (inline slots are `[0, cap)`).
- Keyed on SHA-1 of the full raw prompt (not just a prefix).

## Snapshot Memory Management

### Problem: VRAM Pressure on Discrete GPUs

Each snapshot stores a full copy of the KV cache tensors. For large
contexts this can be substantial:

| Model | max_ctx | Per-snapshot size |
|-------|---------|-------------------|
| Qwen3.5-27B (Q8_0 KV) | 64000 | ~1.3–1.5 GB |
| Gemma4 | 32768 | ~0.5–0.8 GB |
| Laguna | 16384 | ~0.2–0.4 GB |

With `cap=4` slots, this can consume 4–6 GB. On a 22 GB discrete GPU that
already holds model weights (~15 GB) plus runtime cache (~2 GB), this causes
VRAM spill to system RAM via PCIe, resulting in 5×+ slowdowns.

### Solution: Platform-Aware Snapshot Backend

Snapshots are stored on a **snapshot backend** selected at init time based
on the compute backend's memory characteristics:

```cpp
// common/snapshot_backend.h

ggml_backend_t create_snapshot_backend(ggml_backend_t compute_backend);
void free_snapshot_backend(ggml_backend_t snap, ggml_backend_t compute);
```

**Decision logic:**

```
compute_backend's default buffer type is host-accessible?
├── YES (unified memory: Metal, AMD iGPU/HALO)
│   └── return compute_backend  (no copy needed, same physical RAM)
└── NO  (discrete VRAM: CUDA, HIP)
    └── return ggml_backend_cpu_init()  (system RAM, off-GPU)
```

**Platform behavior:**

| Platform | GPU type | Snapshot storage | Copy cost |
|----------|----------|-----------------|-----------|
| CUDA (discrete) | RTX 2080 Ti, etc. | System RAM (CPU backend) | GPU↔CPU via PCIe at save/restore |
| Metal (Apple Silicon) | Unified | Same as compute (no-op) | Zero (same memory) |
| AMD HALO / iGPU | Unified | Same as compute (no-op) | Zero (same memory) |
| HIP (discrete) | RX 7900, etc. | System RAM (CPU backend) | GPU↔CPU via PCIe at save/restore |

### Cross-Backend Transfer

`ggml_backend_tensor_copy(src, dst)` transparently handles:
- Same-backend copy (unified case): direct memcpy or no-op
- Cross-backend copy (discrete case): GPU→CPU get + CPU→GPU set internally

This means snapshot save/restore code is **backend-agnostic** — it just
calls `ggml_backend_tensor_copy()` regardless of where tensors reside.

### Integration Pattern (per-backend)

Each backend adds a single member and three code points:

```cpp
// Header:
ggml_backend_t snap_backend_ = nullptr;

// init():
snap_backend_ = create_snapshot_backend(compute_backend_);

// snapshot_save(): allocate snapshot tensors on snap_backend_
ggml_backend_alloc_ctx_tensors(snap.ctx, snap_backend_);
ggml_backend_tensor_copy(cache.k[il], snap.k[il]);  // works cross-backend

// shutdown(): free in correct order
for (auto & s : snapshots_) free_snapshot(s);  // free tensors first
free_snapshot_backend(snap_backend_, compute_backend_);  // then backend
```

## Configuration

| Server flag | Default | Description |
|-------------|---------|-------------|
| `--prefix-cache-cap N` | 4 | Max inline prefix cache slots |
| `--prefix-cache-full N` | 0 | Max full-compress cache slots |
| `--skip-park` | false | Skip parking draft model during compress |

## File Map

| File | Role |
|------|------|
| `server/prefix_cache.{h,cpp}` | LRU logic, boundary detection, hashing |
| `common/snapshot_backend.h` | Platform-aware snapshot backend selection |
| `common/model_backend.h` | `snapshot_save/free/used/cur_pos` interface |
| `qwen35/qwen35_target_graph.cpp` | `snapshot_target_cache()`, `restore_target_cache()` |
| `laguna/laguna_target_graph.cpp` | `laguna_snapshot_alloc/save/restore()` |
| `gemma4/gemma4_backend.cpp` | Inline snapshot allocation + copy |

## Performance Characteristics

### Save (prefill → snapshot)
- **Unified memory**: Near-instant (just memcpy, no PCIe)
- **Discrete GPU**: PCIe transfer at ~12 GB/s (e.g., 1.5 GB → ~125ms)
- Amortized over the full prefill time (typically seconds for long prompts)

### Restore (snapshot → KV cache)
- **Unified memory**: Near-instant
- **Discrete GPU**: PCIe transfer (~125ms for 1.5 GB)
- Always faster than re-computing prefill (which takes seconds)

### Net Impact
On discrete GPUs, the snapshot restore adds ~100-200ms latency but avoids
repeating a multi-second prefill. VRAM stays free for model weights and
active KV cache, preventing the catastrophic spill-to-system-RAM that
causes 5× decode slowdowns.
