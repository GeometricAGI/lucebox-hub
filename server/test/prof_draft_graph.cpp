// Profiling harness for the DFlash draft forward graph (src/draft/draft_graph.cpp).
//
// Unlike smoke_draft_graph (single shot, correctness only), this:
//   - runs WARMUP + MEASURED iterations so kernel timings are steady-state
//     (first runs include CUDA JIT / autotuning and are discarded),
//   - wraps each measured forward in an NVTX range ("draft_forward") plus an
//     outer "measured" range, so Nsight Systems can isolate the draft region,
//   - reports per-iteration wall latency (min/mean/median/max) using backend
//     synchronisation, so you get an end-to-end number even without a profiler,
//   - walks the built ggml graph and prints a per-op histogram — a
//     profiler-independent view of fusion candidates (how many rms_norm/mul,
//     mul_mat, cont/permute copies, etc. the 5-layer draft emits).
//
// Usage:
//   prof_draft_graph <draft.safetensors> [--ctx-len N] [--iters N] [--warmup N]
//
// Pair with scripts/prof_draft.sh, which runs this under nsys and post-processes
// the kernel/idle/memcpy breakdown.

#include "dflash27b.h"
#include "internal.h"
#include "draft_graph.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

// NVTX is header-only in nvtx3; degrade to no-ops if the header is unavailable
// so the harness always builds.
#if __has_include(<nvtx3/nvToolsExt.h>)
  #include <nvtx3/nvToolsExt.h>
  #define PROF_NVTX_PUSH(name) nvtxRangePushA(name)
  #define PROF_NVTX_POP()      nvtxRangePop()
#elif __has_include(<nvToolsExt.h>)
  #include <nvToolsExt.h>
  #define PROF_NVTX_PUSH(name) nvtxRangePushA(name)
  #define PROF_NVTX_POP()      nvtxRangePop()
#else
  #define PROF_NVTX_PUSH(name) ((void)0)
  #define PROF_NVTX_POP()      ((void)0)
#endif

using namespace dflash::common;

static int arg_int(int argc, char ** argv, const char * flag, int dflt) {
    for (int i = 1; i < argc - 1; i++) {
        if (std::strcmp(argv[i], flag) == 0) return std::atoi(argv[i + 1]);
    }
    return dflt;
}

// Human-readable op label, expanding GGML_OP_UNARY into its specific unary op.
static std::string op_label(const ggml_tensor * node) {
    if (node->op == GGML_OP_UNARY) {
        return std::string("UNARY/") + ggml_unary_op_name(ggml_get_unary_op(node));
    }
    return ggml_op_name(node->op);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
            "usage: %s <model.safetensors> [--ctx-len N] [--iters N] [--warmup N]\n",
            argv[0]);
        return 2;
    }
    const char * path   = argv[1];
    const int ctx_len   = arg_int(argc, argv, "--ctx-len", 512);
    const int iters     = arg_int(argc, argv, "--iters",   50);
    const int warmup    = arg_int(argc, argv, "--warmup",  10);

    const int q_len  = DFLASH27B_DRAFT_BLOCK_SIZE;            // 16
    const int hidden = DFLASH27B_TARGET_HIDDEN;               // 5120
    const int fc_in  = DFLASH27B_DRAFT_N_TARGET_LAYERS * hidden;

    std::printf("[prof] ctx_len=%d q_len=%d hidden=%d fc_in=%d iters=%d warmup=%d\n",
                ctx_len, q_len, hidden, fc_in, iters, warmup);

    // ── Backend + weights
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { std::fprintf(stderr, "ggml_backend_cuda_init failed\n"); return 1; }

    DraftWeights w;
    if (!load_draft_safetensors(path, backend, w)) {
        std::fprintf(stderr, "load: %s\n", dflash27b_last_error());
        return 1;
    }
    std::printf("[prof] draft loaded: n_layer=%d n_head=%d n_head_kv=%d head_dim=%d swa_window=%d\n",
                w.n_layer, w.n_head, w.n_head_kv, w.head_dim, w.swa_window);

    // ── Graph context
    ggml_init_params ip{};
    ip.mem_size   = 256ull * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    ggml_context * gctx = ggml_init(ip);
    if (!gctx) { std::fprintf(stderr, "ggml_init graph failed\n"); return 1; }

    // ── Input placeholders (F32 activations; weights stay bf16)
    ggml_tensor * noise_embed = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, hidden, q_len, 1);
    ggml_tensor * target_hid  = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, fc_in, ctx_len, 1);
    ggml_tensor * pos_q       = ggml_new_tensor_1d(gctx, GGML_TYPE_I32,  q_len);
    ggml_tensor * pos_k       = ggml_new_tensor_1d(gctx, GGML_TYPE_I32,  ctx_len + q_len);
    ggml_set_name(noise_embed, "noise_embed");
    ggml_set_name(target_hid,  "target_hidden_cat");
    ggml_set_name(pos_q,       "positions_q");
    ggml_set_name(pos_k,       "positions_k");
    ggml_set_input(noise_embed);
    ggml_set_input(target_hid);
    ggml_set_input(pos_q);
    ggml_set_input(pos_k);

    const bool any_swa = (w.swa_window > 0) && [&] {
        for (int il = 0; il < w.n_layer; il++) if (w.layers[il].is_swa) return true;
        return false;
    }();
    if (any_swa && ctx_len > w.swa_window) {
        std::printf("[prof] NOTE: model has SWA layers (window=%d) but this harness runs "
                    "with causal_mask_swa=nullptr (all layers non-causal). For SWA models "
                    "supply a mask to profile the real attention path.\n", w.swa_window);
    }

    DraftGraphInputs gi{};
    gi.ctx_len           = ctx_len;
    gi.noise_embed       = noise_embed;
    gi.target_hidden_cat = target_hid;
    gi.positions_q       = pos_q;
    gi.positions_k       = pos_k;

    DraftGraphOutputs go = build_draft_graph(gctx, w, gi);
    if (!go.hidden_states) { std::fprintf(stderr, "build_draft_graph returned null\n"); return 1; }
    ggml_set_output(go.hidden_states);

    ggml_cgraph * gf = ggml_new_graph(gctx);
    ggml_build_forward_expand(gf, go.hidden_states);
    const int n_nodes = ggml_graph_n_nodes(gf);

    // ── Static op histogram: fusion-candidate view, independent of the profiler.
    std::map<std::string, int> op_hist;
    for (int i = 0; i < n_nodes; i++) {
        op_hist[op_label(ggml_graph_node(gf, i))]++;
    }
    std::printf("[prof] graph n_nodes=%d  (per-op counts; whole graph = %d draft layers)\n",
                n_nodes, w.n_layer);
    std::vector<std::pair<std::string,int>> hist(op_hist.begin(), op_hist.end());
    std::sort(hist.begin(), hist.end(), [](auto&a, auto&b){ return a.second > b.second; });
    for (auto & [name, cnt] : hist) {
        std::printf("[prof]   %-22s x%-4d (~%.1f/layer)\n",
                    name.c_str(), cnt, (double)cnt / std::max(1, w.n_layer));
    }

    // ── Allocate + fill inputs once (constant across iterations).
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        std::fprintf(stderr, "ggml_gallocr_alloc_graph failed\n");
        return 1;
    }
    {
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> u(-0.02f, 0.02f);
        std::vector<float> ne((size_t)hidden * q_len);
        for (auto & v : ne) v = u(rng);
        ggml_backend_tensor_set(noise_embed, ne.data(), 0, sizeof(float) * ne.size());
        std::vector<float> th((size_t)fc_in * ctx_len);
        for (auto & v : th) v = u(rng);
        ggml_backend_tensor_set(target_hid, th.data(), 0, sizeof(float) * th.size());
        std::vector<int32_t> pq(q_len);
        for (int i = 0; i < q_len; i++) pq[i] = ctx_len + i;
        ggml_backend_tensor_set(pos_q, pq.data(), 0, sizeof(int32_t) * pq.size());
        std::vector<int32_t> pk(ctx_len + q_len);
        for (int i = 0; i < ctx_len + q_len; i++) pk[i] = i;
        ggml_backend_tensor_set(pos_k, pk.data(), 0, sizeof(int32_t) * pk.size());
    }

    // ── Warmup (discarded): absorbs JIT, autotuning, allocator first-touch.
    PROF_NVTX_PUSH("warmup");
    for (int it = 0; it < warmup; it++) {
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "warmup compute failed\n"); return 1;
        }
    }
    ggml_backend_synchronize(backend);
    PROF_NVTX_POP();

    // ── Measured iterations.
    std::vector<double> ms;
    ms.reserve(iters);
    PROF_NVTX_PUSH("measured");
    for (int it = 0; it < iters; it++) {
        ggml_backend_synchronize(backend);
        auto t0 = std::chrono::steady_clock::now();
        PROF_NVTX_PUSH("draft_forward");
        ggml_status st = ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
        PROF_NVTX_POP();
        auto t1 = std::chrono::steady_clock::now();
        if (st != GGML_STATUS_SUCCESS) { std::fprintf(stderr, "compute failed it=%d\n", it); return 1; }
        ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    PROF_NVTX_POP();

    // ── Validate output once (no NaN/Inf).
    {
        std::vector<float> out(ggml_nelements(go.hidden_states));
        ggml_backend_tensor_get(go.hidden_states, out.data(), 0, sizeof(float) * out.size());
        int bad = 0;
        for (float f : out) if (std::isnan(f) || std::isinf(f)) bad++;
        if (bad) { std::fprintf(stderr, "FAIL: %d non-finite output values\n", bad); return 1; }
    }

    // ── Latency stats.
    std::vector<double> sorted = ms;
    std::sort(sorted.begin(), sorted.end());
    double sum = 0; for (double v : ms) sum += v;
    const double mean = sum / ms.size();
    const double med  = sorted[sorted.size() / 2];
    const double mn   = sorted.front();
    const double mx   = sorted.back();
    std::printf("\n[prof] === draft forward latency over %d measured iters (ctx_len=%d) ===\n",
                iters, ctx_len);
    std::printf("[prof] min=%.3f ms  mean=%.3f ms  median=%.3f ms  max=%.3f ms\n",
                mn, mean, med, mx);
    std::printf("[prof] per-forward emits q_len=%d hidden states -> %.1f draft tok/s (median)\n",
                q_len, q_len * 1000.0 / med);

    ggml_gallocr_free(alloc);
    ggml_free(gctx);
    free_draft_weights(w);
    ggml_backend_free(backend);
    return 0;
}
