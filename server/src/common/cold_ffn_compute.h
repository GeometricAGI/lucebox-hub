// ColdFfnCompute: Direct compute interface for cold expert FFN.
// Bypasses ggml graph dispatch overhead. Shared-memory model (CPU/Halo).
#pragma once

#include "ggml.h"
#include <cstdint>
#include <memory>

namespace dflash::common {

// Per-layer cold weight metadata — raw pointers into shared memory.
struct ColdFfnLayer {
    const void * gate_up_data = nullptr;  // fused [n_cold, n_ff*2, n_embd] quantized
    const void * gate_data = nullptr;     // separate gate [n_cold, n_ff, n_embd]
    const void * up_data = nullptr;       // separate up   [n_cold, n_ff, n_embd]
    const void * down_data = nullptr;     // [n_cold, n_embd, n_ff] quantized

    size_t gate_up_stride = 0;   // bytes between experts in gate_up tensor
    size_t gate_stride = 0;      // bytes between experts in gate tensor
    size_t up_stride = 0;        // bytes between experts in up tensor
    size_t down_stride = 0;      // bytes between experts in down tensor

    ggml_type gate_up_type = GGML_TYPE_Q4_K;  // type for fused gate_up
    ggml_type gate_type = GGML_TYPE_Q4_K;     // type for separate gate
    ggml_type up_type = GGML_TYPE_Q4_K;       // type for separate up
    ggml_type down_type = GGML_TYPE_Q4_K;     // type for down projection
    bool fused_gate_up = false;               // true if gate+up are fused

    // Scale factors (applied after matmul). 1.0 = no scaling.
    float gate_up_scale = 1.0f;
    float gate_scale = 1.0f;
    float up_scale = 1.0f;
    float down_scale = 1.0f;
};

// Abstract compute interface. Implementations: CPU (now), Halo (future).
struct ColdFfnCompute {
    virtual ~ColdFfnCompute() = default;

    // Compute cold expert FFN contributions and accumulate into output.
    // input:   [n_embd] F32 — post-norm hidden state
    // ids:     [n_cold] I32 — local cold expert indices
    // weights: [n_cold] F32 — routing weights for each cold expert
    // output:  [n_embd] F32 — accumulated weighted expert outputs (zeroed by callee)
    virtual void compute(
        const ColdFfnLayer & layer,
        const float * input,
        const int32_t * ids,
        const float * weights,
        int n_cold,
        int n_embd,
        int n_ff,
        float * output) = 0;
};

// Create CPU-based fused cold FFN compute.
std::unique_ptr<ColdFfnCompute> make_cpu_cold_ffn_compute(int n_ff_max);

}  // namespace dflash::common
