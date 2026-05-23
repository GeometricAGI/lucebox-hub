// Smoke test for qwen35moe hybrid storage construction.
//
// Loads qwen35moe, builds a simple per-layer placement (1 hot expert per layer),
// constructs hybrid storage, and validates that hot GPU tensors / cold host
// buffers are materialized.

#include "internal.h"
#include "qwen35moe_expert_placement.h"
#include "qwen35moe_hybrid_storage.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cstdio>
#include <cstdlib>
#include <string>

using namespace dflash::common;

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <qwen35moe.gguf>\n", argv[0]);
        return 2;
    }

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        std::fprintf(stderr, "cuda init failed\n");
        return 1;
    }

    TargetWeights w;
    if (!load_target_gguf(argv[1], backend, w)) {
        std::fprintf(stderr, "load_target_gguf: %s\n", dflash27b_last_error());
        return 1;
    }
    if (!w.is_moe) {
        std::fprintf(stderr, "target is not qwen35moe\n");
        return 1;
    }

    Qwen35MoeExpertPlacement placement;
    placement.n_layer = w.n_layer;
    placement.n_expert = w.n_expert;
    placement.n_expert_used = w.n_expert_used;
    placement.total_hot = w.n_layer;
    placement.hot_counts.assign((size_t)w.n_layer, 1);
    placement.hot_expert_ids.resize((size_t)w.n_layer);
    for (int il = 0; il < w.n_layer; ++il) {
        placement.hot_expert_ids[(size_t)il].push_back(0);
    }

    Qwen35MoeHybridStorage hybrid;
    std::string err;
    if (!build_qwen35moe_hybrid_storage(w, backend, placement, hybrid, &err)) {
        std::fprintf(stderr, "build_qwen35moe_hybrid_storage: %s\n", err.c_str());
        return 1;
    }
    if (!hybrid.matches(w)) {
        std::fprintf(stderr, "hybrid storage does not match weights\n");
        return 1;
    }

    for (int il = 0; il < w.n_layer; ++il) {
        const auto & layer = hybrid.layers[(size_t)il];
        if ((int)layer.hot_expert_ids.size() != 1) {
            std::fprintf(stderr, "layer %d hot_expert_ids size=%zu\n", il, layer.hot_expert_ids.size());
            return 1;
        }
        if (layer.hot_local_by_global.size() != (size_t)w.n_expert ||
            layer.cold_local_by_global.size() != (size_t)w.n_expert) {
            std::fprintf(stderr, "layer %d local maps wrong size\n", il);
            return 1;
        }
        if (layer.fused_gate_up) {
            if (!layer.gate_up_hot || layer.gate_up_cold_bytes.empty()) {
                std::fprintf(stderr, "layer %d fused hot/cold storage missing\n", il);
                return 1;
            }
        } else {
            if (!layer.gate_hot || !layer.up_hot ||
                layer.gate_cold_bytes.empty() || layer.up_cold_bytes.empty()) {
                std::fprintf(stderr, "layer %d split hot/cold storage missing\n", il);
                return 1;
            }
        }
        if (!layer.down_hot || layer.down_cold_bytes.empty()) {
            std::fprintf(stderr, "layer %d down hot/cold storage missing\n", il);
            return 1;
        }
    }

    hybrid = Qwen35MoeHybridStorage{};
    free_target_weights(w);
    ggml_backend_free(backend);
    std::printf("OK\n");
    return 0;
}
