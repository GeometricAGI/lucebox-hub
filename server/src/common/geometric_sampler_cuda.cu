// CUDA port of the sample_logits chain. See geometric_sampler_cuda.h for the contract and
// the CPU/GPU split rationale, and sampler.cpp for the reference CPU chain this
// mirrors: rep_penalty -> freq/pres_penalty -> softmax(temp) -> draw.
// top_p (nucleus) is not implemented here — cfg.top_p in (0,1) falls back to
// the CPU chain (see geometric_sampler_cuda.h).
//
// The per-call workload is one logit row (vocab ~150k). That is small enough
// that a single thread block handles the whole row: it keeps every reduction
// and the inverse-CDF scan in shared memory with no cross-block
// synchronization, which is far simpler and — for one row per token — fast
// enough. (The bandwidth-bound multi-block split used by
// geometric_draft_topk_cuda.cu pays off only for many rows at once.)

#include "geometric_sampler_cuda.h"

#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <unordered_map>

namespace dflash::common {

namespace {

constexpr int kPenaltyBlock = 256;  // threads per block for the elementwise penalty
                                    // kernel; not perf-sensitive (a few hundred
                                    // elements at most), so no device query needed.

// geometric_sample_kernel's per-thread shared-memory footprint: two double
// arrays (sh, s_off) and one int32 array (shi), each sized [blockDim.x].
constexpr size_t kSampleShmemPerThread = 2 * sizeof(double) + sizeof(int32_t);

// Per-device sample-kernel block size (threads per row), cached after the
// first call — same single-threaded-decode-loop assumption as Scratch below.
// Queried rather than hardcoded because maxThreadsPerBlock and
// sharedMemPerBlock vary across GPUs (e.g. older compute-capability parts cap
// at 512 threads and/or less shared memory than the 1024-thread/20-byte-per-
// thread shape that fits an RTX 3090).
struct BlockCfg {
    int device = -1;
    int block  = 0;
};
BlockCfg g_block_cfg;

int pick_block_size(int device) {
    if (g_block_cfg.device == device) return g_block_cfg.block;
    int    max_threads = 1024;
    size_t smem_cap    = 48 * 1024;  // conservative fallback if the query fails
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device) == cudaSuccess) {
        max_threads = prop.maxThreadsPerBlock;
        smem_cap    = prop.sharedMemPerBlock;
    } else {
        cudaGetLastError();
    }
    int block = 1;
    while (block * 2 <= max_threads) block *= 2;  // largest power of two <= max_threads
    // Halve further if the reductions' shared-memory arrays wouldn't fit
    // (stays a power of two so the block-reduction loop below still works).
    while (block > 32 && (size_t)block * kSampleShmemPerThread > smem_cap) block /= 2;
    g_block_cfg = {device, block};
    return block;
}

// Apply the (already-prepared) penalties in place to the working logit copy.
// One thread per affected token id. `rep_inv` is 1/rep_pen folded in by the host
// (or 1.0 to disable); `add[t]` folds freq_pen*count + pres_pen for that token.
// Order matches the CPU chain: multiplicative repetition penalty first, then the
// additive frequency/presence subtraction.
__global__ void geometric_apply_penalties(float * __restrict__ work,
                                const int32_t * __restrict__ ids,
                                const float * __restrict__ add,
                                int m, float rep_pen, int rep_active) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= m) return;
    const int id = ids[t];
    float v = work[id];
    if (rep_active) v = (v > 0.0f) ? v / rep_pen : v * rep_pen;
    v -= add[t];
    work[id] = v;
}

// Dynamic shared memory for geometric_sample_kernel, sized at launch to
// blockDim.x * kSampleShmemPerThread (see pick_block_size). Laid out as two
// double arrays (8-byte aligned, so they come first) then one int32 array.
extern __shared__ unsigned char geometric_smem[];

// Single-block sampler. work[] holds the post-penalty logits. Writes the chosen
// token id to *out. do_sample==0 -> greedy argmax (lowest id wins ties, matching
// the CPU manual-argmax and DFLASH_GPU_ARGMAX behaviour).
__global__ void geometric_sample_kernel(const float * __restrict__ work, int vocab,
                              float inv_t, int do_sample,
                              double r_uniform, int32_t * __restrict__ out) {
    const int t        = threadIdx.x;
    const int nthreads = blockDim.x;
    const int chunk    = (vocab + nthreads - 1) / nthreads;
    const int begin    = t * chunk;
    const int end      = min(begin + chunk, vocab);

    double * sh    = reinterpret_cast<double *>(geometric_smem);
    double * s_off = sh + nthreads;
    int    * shi   = reinterpret_cast<int *>(s_off + nthreads);
    __shared__ float  s_xmax;
    __shared__ int    s_argmax;
    __shared__ double s_scalar;

    // ---- pass 1: max logit + argmax (lowest id on ties) -------------------
    float lmax = -FLT_MAX;
    int   largmax = vocab;  // sentinel so a real id always wins
    for (int i = begin; i < end; i++) {
        const float v = work[i];
        if (v > lmax || (v == lmax && i < largmax)) { lmax = v; largmax = i; }
    }
    sh[t] = lmax; shi[t] = largmax;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (t < s) {
            const double a = sh[t], b = sh[t + s];
            if (b > a || (b == a && shi[t + s] < shi[t])) { sh[t] = b; shi[t] = shi[t + s]; }
        }
        __syncthreads();
    }
    if (t == 0) { s_xmax = (float)sh[0] * inv_t; s_argmax = shi[0]; }
    __syncthreads();

    if (!do_sample) {
        if (t == 0) *out = s_argmax;
        return;
    }
    const float xmax = s_xmax;

    // ---- pass 2: softmax denominator Z = sum exp(x_i - xmax) --------------
    double lz = 0.0;
    for (int i = begin; i < end; i++)
        lz += exp((double)work[i] * inv_t - xmax);
    sh[t] = lz;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (t < s) sh[t] += sh[t + s];
        __syncthreads();
    }
    if (t == 0) s_scalar = sh[0];
    __syncthreads();
    const double Z = s_scalar;

    // ---- multinomial inverse-CDF draw over the full distribution ---------
    // Each thread owns a contiguous id chunk, so ascending-id order is exactly
    // thread 0's chunk, then thread 1's, ... A serial exclusive scan over the
    // nthreads per-thread masses gives each thread its CDF offset; the one whose
    // [offset, offset+mass) straddles the target re-scans its chunk for the id.
    double pm = 0.0;
    for (int i = begin; i < end; i++)
        pm += exp((double)work[i] * inv_t - xmax);
    sh[t] = pm;
    __syncthreads();

    // Thread 0: seed the safety default (used only if fp rounding leaves no
    // straddling thread) and compute the exclusive prefix offsets.
    if (t == 0) {
        *out = s_argmax;
        double acc = 0.0;
        for (int k = 0; k < nthreads; k++) { s_off[k] = acc; acc += sh[k]; }
    }
    __syncthreads();

    const double targetv = r_uniform * Z;
    if (targetv >= s_off[t] && targetv < s_off[t] + pm) {
        double acc = s_off[t];
        for (int i = begin; i < end; i++) {
            acc += exp((double)work[i] * inv_t - xmax);
            if (targetv < acc) { *out = i; break; }
        }
    }
}

// Per-device persistent scratch. The decode loop is single-threaded, so a plain
// static cache avoids a cudaMalloc/cudaFree per token (mirrors geometric_draft_topk_cuda).
struct Scratch {
    int       device   = -1;
    int       vocab_cap = 0;
    int       pen_cap  = 0;
    float *   d_work   = nullptr;  // [vocab] mutable logit copy
    int32_t * d_pen_id = nullptr;  // [pen_cap]
    float *   d_pen_add = nullptr; // [pen_cap]
    int32_t * d_out    = nullptr;  // [1]
};
Scratch g_scratch;

void free_scratch() {
    if (g_scratch.d_work)   cudaFree(g_scratch.d_work);
    if (g_scratch.d_pen_id) cudaFree(g_scratch.d_pen_id);
    if (g_scratch.d_pen_add) cudaFree(g_scratch.d_pen_add);
    if (g_scratch.d_out)    cudaFree(g_scratch.d_out);
    g_scratch = Scratch{};
}

bool ensure_scratch(int device, int vocab, int pen) {
    const bool ok = g_scratch.device == device &&
                    g_scratch.vocab_cap >= vocab &&
                    g_scratch.pen_cap >= pen;
    if (ok) return true;
    free_scratch();
    if (cudaMalloc(&g_scratch.d_out, sizeof(int32_t)) != cudaSuccess) goto fail;
    if (cudaMalloc(&g_scratch.d_work, (size_t)vocab * sizeof(float)) != cudaSuccess) goto fail;
    if (pen > 0) {
        if (cudaMalloc(&g_scratch.d_pen_id, (size_t)pen * sizeof(int32_t)) != cudaSuccess) goto fail;
        if (cudaMalloc(&g_scratch.d_pen_add, (size_t)pen * sizeof(float)) != cudaSuccess) goto fail;
    }
    g_scratch.device    = device;
    g_scratch.vocab_cap = vocab;
    g_scratch.pen_cap   = pen;
    return true;
fail:
    free_scratch();
    return false;
}

}  // namespace

bool gpu_sampler_enabled() {
    static const bool on = []() {
        const char * v = std::getenv("DFLASH_GPU_SAMPLE");
        if (v == nullptr || v[0] == '\0') return true;  // on by default
        return v[0] != '0';                             // "0" (or "0...") opts out
    }();
    return on;
}

int geometric_sample_logits_cuda(const float * logits,
                       int vocab,
                       const SamplerCfg & cfg,
                       const std::vector<int32_t> & history,
                       double r_uniform,
                       bool logits_on_device) {
    if (!logits || vocab <= 0) return -1;
    // top_k stays on the CPU (a single-row partial_sort beats a per-token GPU
    // select); signal fallback.
    if (cfg.top_k > 0 && cfg.top_k < vocab) return -1;
    // top_p (nucleus) is not implemented on this kernel; signal fallback (see
    // geometric_sampler_cuda.h for why).
    if (cfg.top_p > 0.0f && cfg.top_p < 1.0f) return -1;

    // Pick the device. For a device pointer, derive it from the allocation so we
    // run where the logits live; otherwise use the current device.
    int dev = 0;
    if (logits_on_device) {
        cudaPointerAttributes attr{};
        if (cudaPointerGetAttributes(&attr, logits) != cudaSuccess) { cudaGetLastError(); return -1; }
        if (attr.type != cudaMemoryTypeDevice) return -1;
        dev = attr.device;
    } else {
        cudaGetDevice(&dev);
    }
    int prev = 0;
    cudaGetDevice(&prev);
    if (dev != prev) cudaSetDevice(dev);

    int result = -1;

    // ---- penalty index prep on the CPU (history is tiny) -----------------
    // Collect, per unique token in the rep_window, the additive amount
    // (freq_pen*count + pres_pen) and whether the repetition penalty applies.
    std::vector<int32_t> pen_id;
    std::vector<float>   pen_add;
    const bool rep_active  = cfg.rep_pen > 1.0f;
    const bool add_active  = (cfg.freq_pen != 0.0f || cfg.pres_pen != 0.0f);
    if ((rep_active || add_active) && !history.empty()) {
        const int win  = std::min((int)history.size(), cfg.rep_window);
        const int from = (int)history.size() - win;
        std::unordered_map<int, int> counts;
        for (int i = from; i < (int)history.size(); i++) counts[history[i]]++;
        pen_id.reserve(counts.size());
        pen_add.reserve(counts.size());
        for (const auto & kv : counts) {
            if (kv.first < 0 || kv.first >= vocab) continue;
            pen_id.push_back(kv.first);
            pen_add.push_back(add_active ? (cfg.freq_pen * kv.second + cfg.pres_pen) : 0.0f);
        }
    }
    const int m = (int)pen_id.size();

    if (ensure_scratch(dev, vocab, m)) {
        cudaError_t err = cudaSuccess;
        // Get logits into the mutable working copy.
        err = cudaMemcpy(g_scratch.d_work, logits, (size_t)vocab * sizeof(float),
                         logits_on_device ? cudaMemcpyDeviceToDevice
                                          : cudaMemcpyHostToDevice);
        if (err == cudaSuccess && m > 0) {
            cudaMemcpy(g_scratch.d_pen_id, pen_id.data(), (size_t)m * sizeof(int32_t),
                       cudaMemcpyHostToDevice);
            cudaMemcpy(g_scratch.d_pen_add, pen_add.data(), (size_t)m * sizeof(float),
                       cudaMemcpyHostToDevice);
            const int blocks = (m + kPenaltyBlock - 1) / kPenaltyBlock;
            geometric_apply_penalties<<<blocks, kPenaltyBlock>>>(g_scratch.d_work, g_scratch.d_pen_id,
                                                g_scratch.d_pen_add, m, cfg.rep_pen,
                                                rep_active ? 1 : 0);
        }
        if (err == cudaSuccess) {
            const int    do_sample    = (cfg.temp > 0.0f) ? 1 : 0;
            const float  inv_t        = 1.0f / fmaxf(1e-3f, cfg.temp);
            const int    block        = pick_block_size(dev);
            const size_t shmem_bytes  = (size_t)block * kSampleShmemPerThread;
            geometric_sample_kernel<<<1, block, shmem_bytes>>>(g_scratch.d_work, vocab, inv_t, do_sample,
                                         r_uniform, g_scratch.d_out);
            int32_t tok = -1;
            if (cudaGetLastError() == cudaSuccess &&
                cudaMemcpy(&tok, g_scratch.d_out, sizeof(int32_t),
                           cudaMemcpyDeviceToHost) == cudaSuccess) {
                result = tok;
            }
        }
    }
    if (result < 0) cudaGetLastError();  // clear so we don't poison the next call
    if (dev != prev) cudaSetDevice(prev);
    return result;
}

}  // namespace dflash::common
