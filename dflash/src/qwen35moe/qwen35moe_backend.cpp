#include "qwen35moe_backend.h"

#include <cstdio>
#include <cstdlib>

namespace dflash::common {

Qwen35MoeBackend::Qwen35MoeBackend(const Qwen35Config & cfg)
    : Qwen35Backend(cfg) {}

bool Qwen35MoeBackend::load_target_model(ggml_backend_t backend, TargetWeights & out) {
    if (!load_target_gguf(cfg_.target_path, backend, out)) {
        return false;
    }

    const char * placement_path = std::getenv("DFLASH_QWEN35MOE_PLACEMENT");
    if (!placement_path || !placement_path[0]) {
        return true;
    }

    Qwen35MoeExpertPlacement placement;
    std::string err;
    if (!Qwen35MoeExpertPlacement::load_json(placement_path, placement, &err)) {
        set_last_error(std::string("qwen35moe placement load failed: ") + err);
        return false;
    }

    auto hybrid = std::make_shared<Qwen35MoeHybridStorage>();
    if (!build_qwen35moe_hybrid_storage(out, backend, placement, *hybrid, &err)) {
        set_last_error(std::string("qwen35moe hybrid storage build failed: ") + err);
        return false;
    }
    out.moe_hybrid = std::move(hybrid);
    hybrid_mode_ = true;
    cfg_.draft_path = nullptr;  // policy: hybrid mode falls back to AR-only until hybrid FFN lands
    int total_cold = 0;
    for (const auto & layer : out.moe_hybrid->layers) {
        total_cold += (int)layer.cold_expert_ids.size();
    }
    std::printf("[qwen35moe] hybrid storage ready: total_hot=%d total_cold=%d placement=%s (AR-only mode)\n",
                out.moe_hybrid->placement.total_hot, total_cold, placement_path);
    return true;
}

}  // namespace dflash::common
