# Hybrid Rollback for Qwen3.6-27B DFlash

## Background

Qwen3.6-27B is a hybrid architecture with a **3:1 ratio of GatedDeltanetAttention to
GatedAttention** layers. Each layer type has a fundamentally different state structure,
and the current rollback strategy treats them uniformly — which is correct but wasteful.

## Current rollback paths

After `verify_batch` runs speculatively over `q_len=16` draft tokens, the target's
internal state reflects all 16 positions. If only `accept_n` tokens were accepted,
the state must be wound back to `committed + accept_n`. There are currently two paths:

### Fast rollback (when `accept_n >= 5`)

`verify_batch` was called with `capture_ssm_intermediates=true`, so the GatedDeltanet
recurrent state at every intermediate position is already in memory.
`rollback_to(committed, commit_n)` copies the saved state at the accepted position back
as the live state. No second forward pass needed.

**Cost still paid:** the GatedAttention KV cache was also updated speculatively for all 16
positions. Fast rollback currently handles this too (either by truncating or restoring from
the pre-verify snapshot, depending on implementation), so there is some overhead even on
the happy path.

### Legacy restore+replay (fallback or `accept_n < 5`)

```
restore_kv()                           // revert to pre-verify snapshot
verify_batch(replay_tok, ..., nullptr) // re-run forward on accepted tokens only
```

This is correct for both layer types but pays the full cost of a second forward pass
over `commit_n` tokens.

## The problem: treating both layer types the same

The two layer types have different rollback costs:

| Layer type              | Count | State structure | Rollback cost |
|-------------------------|:-----:|-----------------|---------------|
| GatedAttention          |  1/4  | KV cache: independent per-position rows | Truncate to `committed + accept_n` — O(1), free |
| GatedDeltanetAttention  |  3/4  | Recurrent state matrix: cumulative over all processed tokens | Cannot truncate; must restore from a saved intermediate |

The KV cache rows written during speculative verify are **positionally independent** —
row at position `i` depends only on the token at position `i` and does not change when
later tokens are processed. Truncating the cache to `committed + accept_n` is just a
length/pointer update; there is nothing to undo.

The GatedDeltanet recurrent state is the opposite — it is updated multiplicatively by
every token, so after processing 16 draft tokens it encodes information from all 16.
You cannot surgically remove the last `16 - accept_n` contributions.

## Proposed hybrid rollback

Combine the two approaches, one per layer type:

1. **GatedAttention layers:** after acceptance, truncate KV cache to `committed + accept_n`.
   No snapshot before verify, no restore after. Zero overhead on the acceptance hot path.

2. **GatedDeltanetAttention layers:** continue capturing recurrent state intermediates
   during `verify_batch` and use `rollback_to` to restore the state at the accepted
   position, exactly as fast rollback does today.

Together this **eliminates the replay pass entirely** — even on the fallback path.
The `restore_kv` + second `verify_batch` call exists solely because there was no cheaper
way to get the attention KV and GatedDeltanet state simultaneously correct. With
per-layer-type rollback, both are handled directly:

```
verify_batch(draft_tok, ..., capture_ssm_intermediates=true)
// acceptance logic → accept_n, commit_n

// GatedAttention: truncate KV tail
attention_cache.truncate(committed + commit_n)          // O(1)

// GatedDeltanet: restore recurrent state from captured intermediate
target.rollback_to(committed, commit_n)                 // existing fast-rollback path
```

## Expected wins

### Eliminate snapshot overhead
`snapshot_kv()` is called before every `verify_batch`. For the attention layers this
snapshot is no longer needed (truncation replaces it). Only the GatedDeltanet
intermediates need capturing, which already happens inside `verify_batch` via the
`capture_ssm_intermediates` flag.

### Eliminate the replay pass
The second `verify_batch(replay_tok, ...)` call currently runs on every step where
`accept_n < 5` (the fast-rollback threshold). For Qwen3.6-27B this is a significant
fraction of steps — the current drafter is still under training and produces lower
AL than the Qwen3.5 drafter (AL ~5.32 vs ~8.31 on HumanEval on 4090; see RESULTS.md).
At AL 5.32 the threshold of 5 is right at the boundary, meaning many steps fall through
to legacy replay. Removing the threshold condition entirely eliminates this cost.

### Reduce intermediate capture footprint
Currently `capture_ssm_intermediates` captures state for both layer types implicitly.
If GatedAttention layers no longer participate in rollback, only the GatedDeltanet
recurrent state (3/4 of layers) needs to be stored, reducing per-step memory and
bandwidth by ~25%.

## Implementation notes

- `rollback_to` in `qwen35_dflash_target.cpp` must be audited: confirm it handles
  GatedDeltanet state independently of the attention KV, and that it does not
  currently also restore the attention KV (which would need to be split out).
- Attention KV truncation needs a lightweight path: decrement the cached sequence
  length to `committed + commit_n` without zeroing or freeing the tail — those slots
  will be overwritten in the next step.
- `snapshot_kv` / `restore_kv` can be removed from the hot path once attention
  truncation is in place. Keep as a fallback for error recovery.
- The `kFastRollbackThreshold = 5` condition in `dflash_spec_decode.cpp:218` can be
  removed — hybrid rollback is always correct and always cheaper than replay.