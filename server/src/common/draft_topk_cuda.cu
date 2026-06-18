// GPU top-K + log-prob extraction for DDTree draft distributions.
// See draft_topk_cuda.h for the contract. Mirrors extract_draft_topk (ddtree.cpp).

#include "draft_topk_cuda.h"

#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace dflash::common {

namespace {

constexpr int kMaxK  = 8;     // ddtree_K is 8 in practice; K>kMaxK → CPU fallback
constexpr int kBlock = 1024;  // threads per position (power of two for the reduction)
// With only n_positions (~15) blocks, occupancy is capped by blocks, not
// threads — so we want many warps per block to hide the vocab read latency.
// 1024 threads × kMaxK × (float+int32) = 64 KiB dynamic shared, which exceeds
// the 48 KiB default and needs an opt-in (see ensure_smem_optin below).

// One block per draft position. A single strided pass over the vocabulary
// accumulates a per-thread online logsumexp (running max + sum) and a
// per-thread sorted-descending top-K; a shared-memory tree reduction then
// merges both across threads. log_prob[k] = scaled_logit[k] - log_z.
__global__ void draft_topk_kernel(const float * __restrict__ logits,
                                  int vocab, int K, float inv_t,
                                  float * __restrict__ out_lp,
                                  int32_t * __restrict__ out_ids) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float * __restrict__ li = logits + (size_t)row * vocab;

    float lmax = -FLT_MAX;
    float lsum = 0.0f;
    float topv[kMaxK];
    int   topi[kMaxK];
#pragma unroll
    for (int k = 0; k < kMaxK; k++) { topv[k] = -FLT_MAX; topi[k] = -1; }

    // ---- single strided pass: logsumexp + local top-K ------------------
    // j ascends within a thread, so a strict-greater insert keeps the lower
    // id on ties (matches the CPU heap's first-wins behaviour).
    for (int j = tid; j < vocab; j += kBlock) {
        const float l = li[j] * inv_t;
        if (l > lmax) { lsum = lsum * __expf(lmax - l) + 1.0f; lmax = l; }
        else          { lsum += __expf(l - lmax); }
        if (l > topv[K - 1]) {
            int p = K - 1;
            while (p > 0 && topv[p - 1] < l) {
                topv[p] = topv[p - 1]; topi[p] = topi[p - 1]; --p;
            }
            topv[p] = l; topi[p] = j;
        }
    }

    // ---- block reduction --------------------------------------------------
    __shared__ float s_max[kBlock];
    __shared__ float s_sum[kBlock];
    extern __shared__ char s_raw[];
    float   * s_topv = reinterpret_cast<float   *>(s_raw);
    int32_t * s_topi = reinterpret_cast<int32_t *>(s_topv + (size_t)kBlock * K);

    s_max[tid] = lmax;
    s_sum[tid] = lsum;
    for (int k = 0; k < K; k++) {
        s_topv[(size_t)tid * K + k] = topv[k];
        s_topi[(size_t)tid * K + k] = topi[k];
    }
    __syncthreads();

    for (int stride = kBlock / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            // logsumexp merge of (max,sum) pairs
            const float am = s_max[tid],          as = s_sum[tid];
            const float bm = s_max[tid + stride], bs = s_sum[tid + stride];
            const float m = fmaxf(am, bm);
            s_sum[tid] = as * __expf(am - m) + bs * __expf(bm - m);
            s_max[tid] = m;

            // merge two sorted-desc top-K lists, keep the K largest (lower id on tie)
            float av[kMaxK]; int ai[kMaxK];
            float bv[kMaxK]; int bi[kMaxK];
            for (int k = 0; k < K; k++) {
                av[k] = s_topv[(size_t)tid * K + k];
                ai[k] = s_topi[(size_t)tid * K + k];
                bv[k] = s_topv[(size_t)(tid + stride) * K + k];
                bi[k] = s_topi[(size_t)(tid + stride) * K + k];
            }
            int ia = 0, ib = 0;
            for (int k = 0; k < K; k++) {
                bool takeA;
                if      (ib >= K)          takeA = true;
                else if (ia >= K)          takeA = false;
                else if (av[ia] > bv[ib])  takeA = true;
                else if (av[ia] < bv[ib])  takeA = false;
                else                       takeA = (ai[ia] <= bi[ib]);  // tie → lower id
                if (takeA) { s_topv[(size_t)tid * K + k] = av[ia]; s_topi[(size_t)tid * K + k] = ai[ia]; ia++; }
                else       { s_topv[(size_t)tid * K + k] = bv[ib]; s_topi[(size_t)tid * K + k] = bi[ib]; ib++; }
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        const float log_z = s_max[0] + logf(s_sum[0]);
        for (int k = 0; k < K; k++) {
            out_lp[(size_t)row * K + k]  = s_topv[k] - log_z;
            out_ids[(size_t)row * K + k] = s_topi[k];
        }
    }
}

// Per-device scratch for the [n_positions × K] outputs, grown as needed. The
// decode loop is single-threaded, so a plain static cache is safe and avoids a
// cudaMalloc/cudaFree on every step.
struct Scratch {
    int       device = -1;
    size_t    cap    = 0;       // elements
    float *   d_lp   = nullptr;
    int32_t * d_ids  = nullptr;
};
Scratch g_scratch;

bool ensure_scratch(int device, size_t n) {
    if (g_scratch.device == device && g_scratch.cap >= n) return true;
    if (g_scratch.d_lp)  cudaFree(g_scratch.d_lp);
    if (g_scratch.d_ids) cudaFree(g_scratch.d_ids);
    g_scratch = Scratch{};
    if (cudaMalloc(&g_scratch.d_lp,  n * sizeof(float))   != cudaSuccess) return false;
    if (cudaMalloc(&g_scratch.d_ids, n * sizeof(int32_t)) != cudaSuccess) {
        cudaFree(g_scratch.d_lp); g_scratch.d_lp = nullptr; return false;
    }
    g_scratch.device = device;
    g_scratch.cap    = n;
    return true;
}

}  // namespace

bool extract_draft_topk_cuda(const void * d_logits,
                             int n_positions, int vocab, int K,
                             float * out_log_probs,
                             int32_t * out_token_ids,
                             float temperature) {
    if (!d_logits || n_positions <= 0 || vocab <= 0 || K <= 0 || K > kMaxK) return false;

    cudaPointerAttributes attr{};
    if (cudaPointerGetAttributes(&attr, d_logits) != cudaSuccess) {
        cudaGetLastError();  // clear the error so we don't poison the next CUDA call
        return false;
    }
    if (attr.type != cudaMemoryTypeDevice) return false;

    int prev = 0;
    cudaGetDevice(&prev);
    const int dev = attr.device;
    if (dev != prev) cudaSetDevice(dev);

    static const bool kProfile = std::getenv("DFLASH_TOPK_PROFILE") != nullptr;
    bool ok = false;
    const size_t n = (size_t)n_positions * K;
    if (ensure_scratch(dev, n)) {
        const float  inv_t = 1.0f / fmaxf(1e-3f, temperature);
        const size_t smem  = (size_t)kBlock * K * (sizeof(float) + sizeof(int32_t));
        // Opt in to >48 KiB dynamic shared (once per device). If it fails the
        // launch below will error and we fall back to the CPU path.
        static int s_optin_dev = -1;
        if (s_optin_dev != dev) {
            constexpr int kMaxSmem = (int)((size_t)kBlock * kMaxK *
                                           (sizeof(float) + sizeof(int32_t)));
            cudaFuncSetAttribute(draft_topk_kernel,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, kMaxSmem);
            s_optin_dev = dev;
        }
        cudaEvent_t e_k0, e_k1, e_c1;
        if (kProfile) { cudaEventCreate(&e_k0); cudaEventCreate(&e_k1); cudaEventCreate(&e_c1); cudaEventRecord(e_k0); }
        draft_topk_kernel<<<n_positions, kBlock, smem>>>(
            static_cast<const float *>(d_logits), vocab, K, inv_t,
            g_scratch.d_lp, g_scratch.d_ids);
        if (kProfile) cudaEventRecord(e_k1);
        if (cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess) {
            const cudaError_t e1 = cudaMemcpy(out_log_probs, g_scratch.d_lp,
                                              n * sizeof(float), cudaMemcpyDeviceToHost);
            const cudaError_t e2 = cudaMemcpy(out_token_ids, g_scratch.d_ids,
                                              n * sizeof(int32_t), cudaMemcpyDeviceToHost);
            ok = (e1 == cudaSuccess && e2 == cudaSuccess);
        }
        if (kProfile) {
            cudaEventRecord(e_c1); cudaEventSynchronize(e_c1);
            float k_ms = 0, c_ms = 0;
            cudaEventElapsedTime(&k_ms, e_k0, e_k1);
            cudaEventElapsedTime(&c_ms, e_k1, e_c1);
            std::fprintf(stderr, "[topk] kernel=%.3f ms  sync+copy=%.3f ms (n_pos=%d vocab=%d)\n",
                         k_ms, c_ms, n_positions, vocab);
            cudaEventDestroy(e_k0); cudaEventDestroy(e_k1); cudaEventDestroy(e_c1);
        }
    }
    if (!ok) cudaGetLastError();
    if (dev != prev) cudaSetDevice(prev);
    return ok;
}

}  // namespace dflash::common
