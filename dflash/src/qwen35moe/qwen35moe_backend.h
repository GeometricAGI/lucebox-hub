// Thin qwen35moe backend wrapper over the shared qwen35-family runtime.

#pragma once

#include "qwen35_backend.h"
#include "qwen35moe_hybrid_storage.h"

namespace dflash::common {

class Qwen35MoeBackend : public Qwen35Backend {
public:
    explicit Qwen35MoeBackend(const Qwen35Config & cfg);
    ~Qwen35MoeBackend() override = default;

    bool supports_dflash_spec_decode() const override { return !hybrid_mode_; }

protected:
    bool load_target_model(ggml_backend_t backend, TargetWeights & out) override;

private:
    bool hybrid_mode_ = false;
};

}  // namespace dflash::common
