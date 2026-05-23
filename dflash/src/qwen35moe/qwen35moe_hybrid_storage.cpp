#include "qwen35moe_hybrid_storage.h"

#include <algorithm>
#include <cstring>

namespace dflash::common {

namespace {

static bool read_expert_slices(ggml_backend_t backend,
                               ggml_tensor * tensor,
                               const std::vector<int32_t> & expert_ids,
                               size_t expert_bytes,
                               std::vector<uint8_t> & out,
                               std::string * err) {
    if (!tensor || expert_ids.empty() || expert_bytes == 0) {
        out.clear();
        return true;
    }
    out.resize(expert_bytes * expert_ids.size());
    for (size_t i = 0; i < expert_ids.size(); ++i) {
        const int32_t expert_id = expert_ids[i];
        const size_t offset = expert_bytes * (size_t)expert_id;
        ggml_backend_tensor_get(tensor, out.data() + expert_bytes * i, offset, expert_bytes);
    }
    (void)backend;
    (void)err;
    return true;
}

static bool validate_expert_tensor(ggml_tensor * tensor, int n_expert, size_t * expert_bytes, std::string * err) {
    if (!tensor) {
        *expert_bytes = 0;
        return true;
    }
    if (tensor->ne[2] != n_expert) {
        if (err) *err = "tensor expert dimension mismatch";
        return false;
    }
    if ((int64_t)tensor->nb[2] <= 0) {
        if (err) *err = "tensor expert stride invalid";
        return false;
    }
    *expert_bytes = (size_t)tensor->nb[2];
    return true;
}

static ggml_tensor * new_like_with_expert_count(ggml_context * ctx, ggml_tensor * src, int hot_count) {
    if (!src || hot_count <= 0) return nullptr;
    return ggml_new_tensor_3d(ctx, src->type, src->ne[0], src->ne[1], hot_count);
}

} // namespace

Qwen35MoeHybridStorage::~Qwen35MoeHybridStorage() {
    for (auto & layer : layers) {
        if (layer.hot_buf) {
            ggml_backend_buffer_free(layer.hot_buf);
            layer.hot_buf = nullptr;
        }
        if (layer.hot_ctx) {
            ggml_free(layer.hot_ctx);
            layer.hot_ctx = nullptr;
        }
        layer.gate_hot = nullptr;
        layer.up_hot = nullptr;
        layer.down_hot = nullptr;
        layer.gate_up_hot = nullptr;
    }
}

bool Qwen35MoeHybridStorage::matches(const TargetWeights & w) const {
    return placement.matches(w) && (int)layers.size() == w.n_layer;
}

bool Qwen35MoeHybridStorage::empty() const {
    return layers.empty();
}

bool build_qwen35moe_hybrid_storage(const TargetWeights & w,
                                    ggml_backend_t backend,
                                    const Qwen35MoeExpertPlacement & placement,
                                    Qwen35MoeHybridStorage & out,
                                    std::string * err) {
    if (!placement.matches(w)) {
        if (err) *err = "placement does not match model";
        return false;
    }
    if (!w.is_moe) {
        if (err) *err = "target is not qwen35moe";
        return false;
    }

    Qwen35MoeHybridStorage tmp;
    tmp.placement = placement;
    tmp.layers.resize((size_t)w.n_layer);

    for (int il = 0; il < w.n_layer; ++il) {
        const TargetLayer & L = w.layers[(size_t)il];
        Qwen35MoeHybridLayerStorage & dst = tmp.layers[(size_t)il];
        dst.hot_expert_ids = placement.hot_expert_ids[(size_t)il];
        dst.hot_local_by_global.assign((size_t)w.n_expert, -1);
        dst.cold_local_by_global.assign((size_t)w.n_expert, -1);

        std::vector<uint8_t> is_hot((size_t)w.n_expert, 0);
        for (size_t i = 0; i < dst.hot_expert_ids.size(); ++i) {
            const int32_t expert = dst.hot_expert_ids[i];
            if (expert < 0 || expert >= w.n_expert) {
                if (err) *err = "hot expert id out of range";
                return false;
            }
            dst.hot_local_by_global[(size_t)expert] = (int32_t)i;
            is_hot[(size_t)expert] = 1;
        }
        for (int expert = 0; expert < w.n_expert; ++expert) {
            if (!is_hot[(size_t)expert]) {
                dst.cold_local_by_global[(size_t)expert] = (int32_t)dst.cold_expert_ids.size();
                dst.cold_expert_ids.push_back((int32_t)expert);
            }
        }

        dst.fused_gate_up = (L.ffn_gate_up_exps != nullptr);
        if (!validate_expert_tensor(L.ffn_gate_exps, w.n_expert, &dst.gate_expert_bytes, err) ||
            !validate_expert_tensor(L.ffn_up_exps, w.n_expert, &dst.up_expert_bytes, err) ||
            !validate_expert_tensor(L.ffn_down_exps, w.n_expert, &dst.down_expert_bytes, err) ||
            !validate_expert_tensor(L.ffn_gate_up_exps, w.n_expert, &dst.gate_up_expert_bytes, err)) {
            return false;
        }

        const int hot_count = (int)dst.hot_expert_ids.size();
        if (hot_count > 0) {
            ggml_init_params ip{};
            ip.mem_size   = 16 * ggml_tensor_overhead();
            ip.mem_buffer = nullptr;
            ip.no_alloc   = true;
            dst.hot_ctx = ggml_init(ip);
            if (!dst.hot_ctx) {
                if (err) *err = "failed to init hot_ctx";
                return false;
            }
            if (dst.fused_gate_up) {
                dst.gate_up_hot = new_like_with_expert_count(dst.hot_ctx, L.ffn_gate_up_exps, hot_count);
                dst.down_hot    = new_like_with_expert_count(dst.hot_ctx, L.ffn_down_exps, hot_count);
            } else {
                dst.gate_hot = new_like_with_expert_count(dst.hot_ctx, L.ffn_gate_exps, hot_count);
                dst.up_hot   = new_like_with_expert_count(dst.hot_ctx, L.ffn_up_exps, hot_count);
                dst.down_hot = new_like_with_expert_count(dst.hot_ctx, L.ffn_down_exps, hot_count);
            }
            dst.hot_buf = ggml_backend_alloc_ctx_tensors(dst.hot_ctx, backend);
            if (!dst.hot_buf) {
                if (err) *err = "failed to allocate hot expert buffer";
                return false;
            }

            auto copy_hot = [&](ggml_tensor * src, ggml_tensor * dst_tensor, size_t expert_bytes) {
                if (!src || !dst_tensor || expert_bytes == 0) return true;
                std::vector<uint8_t> tmp_bytes(expert_bytes);
                for (size_t i = 0; i < dst.hot_expert_ids.size(); ++i) {
                    const int32_t expert_id = dst.hot_expert_ids[i];
                    ggml_backend_tensor_get(src, tmp_bytes.data(), expert_bytes * (size_t)expert_id, expert_bytes);
                    ggml_backend_tensor_set(dst_tensor, tmp_bytes.data(), expert_bytes * i, expert_bytes);
                }
                return true;
            };
            if (dst.fused_gate_up) {
                copy_hot(L.ffn_gate_up_exps, dst.gate_up_hot, dst.gate_up_expert_bytes);
                copy_hot(L.ffn_down_exps, dst.down_hot, dst.down_expert_bytes);
            } else {
                copy_hot(L.ffn_gate_exps, dst.gate_hot, dst.gate_expert_bytes);
                copy_hot(L.ffn_up_exps, dst.up_hot, dst.up_expert_bytes);
                copy_hot(L.ffn_down_exps, dst.down_hot, dst.down_expert_bytes);
            }
        }

        if (dst.fused_gate_up) {
            if (!read_expert_slices(backend, L.ffn_gate_up_exps, dst.cold_expert_ids,
                                    dst.gate_up_expert_bytes, dst.gate_up_cold_bytes, err)) {
                return false;
            }
        } else {
            if (!read_expert_slices(backend, L.ffn_gate_exps, dst.cold_expert_ids,
                                    dst.gate_expert_bytes, dst.gate_cold_bytes, err) ||
                !read_expert_slices(backend, L.ffn_up_exps, dst.cold_expert_ids,
                                    dst.up_expert_bytes, dst.up_cold_bytes, err)) {
                return false;
            }
        }
        if (!read_expert_slices(backend, L.ffn_down_exps, dst.cold_expert_ids,
                                dst.down_expert_bytes, dst.down_cold_bytes, err)) {
            return false;
        }
    }

    out = std::move(tmp);
    return true;
}

}  // namespace dflash::common
