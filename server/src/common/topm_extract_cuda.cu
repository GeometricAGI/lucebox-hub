// Fast GPU top-M candidate extractor for the candidate-restricted LM head.
//
// Given the draft logits [vocab × n_tokens] (column p = position p's vocab
// logits, contiguous), produce cand_ids [M × n_tokens]: M candidate vocab ids
// per position that contain (with high probability) the target's argmax. Used
// to feed restricted_lm_head_q6k. The draft top-M is far cheaper to find
// approximately than exactly: we radix-threshold on the order-preserving uint
// mapping of the float logits via one histogram pass, pick the threshold bin
// whose cumulative top-down count first reaches M, then gather. Bins strictly
// above the threshold are always kept; the threshold bin fills the remainder
// (its intra-bin order is irrelevant — the bin spans <0.05% of the value
// range, so coverage of the true top-M is preserved). Reads the logits twice.

#include "topm_extract_cuda.h"

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

namespace dflash::common {

namespace {

constexpr int kBits = 16;                 // histogram key bits (16 ⇒ ~exact top-M)
constexpr int kBins = 1 << kBits;
constexpr int kBlock = 256;
constexpr int kThrThreads = 256;          // threads for the parallel threshold scan
static_assert(kBins % kThrThreads == 0, "kBins must divide threshold threads");

// Order-preserving map float→uint32: larger float ⇒ larger uint (radix-sort key).
__device__ __forceinline__ uint32_t order_key(float f) {
    uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// Pass 1: per-position histogram of the top kBits of order_key over the vocab.
__global__ void histogram_kernel(const float * __restrict__ logits,
                                 int vocab, int n_tokens,
                                 unsigned int * __restrict__ hist) {  // [n_tokens][kBins]
    const int p = blockIdx.y;
    unsigned int * h = hist + (size_t)p * kBins;
    const float * __restrict__ col = logits + (size_t)p * vocab;
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < vocab;
         v += gridDim.x * blockDim.x) {
        atomicAdd(&h[order_key(col[v]) >> (32 - kBits)], 1u);
    }
}

// Pass 2: find threshold_bin[p] = the bin where the top-down cumulative count
// first reaches M, and above_cnt[p] = count strictly above it (< M). One block
// per position: each thread owns a contiguous chunk of bins, computes its chunk
// total, thread 0 builds the per-chunk "count above" prefix, the single thread
// whose chunk straddles the M-crossing rescans just its chunk. O(kBins/T + T).
__global__ void threshold_kernel(const unsigned int * __restrict__ hist,
                                 int n_tokens, int M,
                                 int * __restrict__ threshold_bin,
                                 unsigned int * __restrict__ above_cnt) {
    const int p = blockIdx.x;
    if (p >= n_tokens) return;
    const unsigned int * h = hist + (size_t)p * kBins;
    const int t = threadIdx.x;
    const int chunk = kBins / kThrThreads;
    const int lo = t * chunk, hi = lo + chunk;       // bins [lo, hi)

    __shared__ unsigned int part[kThrThreads];       // chunk totals
    __shared__ unsigned int above[kThrThreads];      // count strictly above chunk t
    unsigned int s = 0;
    for (int b = lo; b < hi; b++) s += h[b];
    part[t] = s;
    __syncthreads();

    if (t == 0) {                                    // suffix sums over chunks
        unsigned int acc = 0;
        for (int c = kThrThreads - 1; c >= 0; c--) { above[c] = acc; acc += part[c]; }
    }
    __syncthreads();

    // The crossing chunk: count above it < M, but adding its total reaches M.
    if (above[t] < (unsigned)M && (unsigned)M <= above[t] + part[t]) {
        unsigned int cum = above[t];
        int bb = lo;
        for (int b = hi - 1; b >= lo; b--) {
            if (cum + h[b] >= (unsigned)M) { bb = b; break; }
            cum += h[b];
        }
        threshold_bin[p] = bb;
        above_cnt[p]     = cum;                      // count strictly above bb (< M)
    }
}

// Pass 3a: emit all ids in bins strictly above the threshold bin (guaranteed
// in the top-M). Uses a per-position fill counter.
__global__ void gather_above_kernel(const float * __restrict__ logits,
                                    int vocab, int n_tokens, int M,
                                    const int * __restrict__ threshold_bin,
                                    int32_t * __restrict__ cand_ids,
                                    unsigned int * __restrict__ fill) {
    const int p  = blockIdx.y;
    const int tb = threshold_bin[p];
    const float * __restrict__ col = logits + (size_t)p * vocab;
    int32_t * out = cand_ids + (size_t)p * M;
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < vocab;
         v += gridDim.x * blockDim.x) {
        if ((int)(order_key(col[v]) >> (32 - kBits)) > tb) {
            const unsigned int slot = atomicAdd(&fill[p], 1u);
            if (slot < (unsigned)M) out[slot] = v;
        }
    }
}

// Pass 3b: fill the remaining [above_cnt, M) slots from the threshold bin.
__global__ void gather_fill_kernel(const float * __restrict__ logits,
                                   int vocab, int n_tokens, int M,
                                   const int * __restrict__ threshold_bin,
                                   int32_t * __restrict__ cand_ids,
                                   unsigned int * __restrict__ fill) {
    const int p  = blockIdx.y;
    const int tb = threshold_bin[p];
    const float * __restrict__ col = logits + (size_t)p * vocab;
    int32_t * out = cand_ids + (size_t)p * M;
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < vocab;
         v += gridDim.x * blockDim.x) {
        if ((int)(order_key(col[v]) >> (32 - kBits)) == tb) {
            const unsigned int slot = atomicAdd(&fill[p], 1u);
            if (slot < (unsigned)M) out[slot] = v;
            else return;  // this position is full
        }
    }
}

}  // namespace

bool extract_topm_cuda(const float * d_logits, int vocab, int n_tokens, int M,
                       int32_t * d_cand_ids, void * d_scratch, cudaStream_t stream) {
    if (vocab <= 0 || n_tokens <= 0 || M <= 0 || M > vocab) return false;

    // scratch layout: hist[n_tokens*kBins] u32 | threshold_bin[n_tokens] i32 |
    //                 above_cnt[n_tokens] u32 | fill[n_tokens] u32
    auto * base = static_cast<unsigned char *>(d_scratch);
    auto * hist          = reinterpret_cast<unsigned int *>(base);
    auto * threshold_bin = reinterpret_cast<int *>(hist + (size_t)n_tokens * kBins);
    auto * above_cnt     = reinterpret_cast<unsigned int *>(threshold_bin + n_tokens);
    auto * fill          = above_cnt + n_tokens;

    if (cudaMemsetAsync(hist, 0, (size_t)n_tokens * kBins * sizeof(unsigned int), stream) != cudaSuccess)
        return false;
    cudaMemsetAsync(fill, 0, (size_t)n_tokens * sizeof(unsigned int), stream);

    const int nsplit = 64;  // blocks per position for the vocab scan
    dim3 grid(nsplit, n_tokens);
    histogram_kernel<<<grid, kBlock, 0, stream>>>(d_logits, vocab, n_tokens, hist);
    threshold_kernel<<<n_tokens, kThrThreads, 0, stream>>>(
        hist, n_tokens, M, threshold_bin, above_cnt);
    gather_above_kernel<<<grid, kBlock, 0, stream>>>(
        d_logits, vocab, n_tokens, M, threshold_bin, d_cand_ids, fill);
    // Reset fill to above_cnt so 3b continues filling from the right slot.
    cudaMemcpyAsync(fill, above_cnt, (size_t)n_tokens * sizeof(unsigned int),
                    cudaMemcpyDeviceToDevice, stream);
    gather_fill_kernel<<<grid, kBlock, 0, stream>>>(
        d_logits, vocab, n_tokens, M, threshold_bin, d_cand_ids, fill);

    return cudaGetLastError() == cudaSuccess;
}

size_t extract_topm_scratch_bytes(int n_tokens) {
    return (size_t)n_tokens * kBins * sizeof(unsigned int)   // hist
         + (size_t)n_tokens * sizeof(int)                     // threshold_bin
         + (size_t)n_tokens * sizeof(unsigned int)            // above_cnt
         + (size_t)n_tokens * sizeof(unsigned int);           // fill
}

}  // namespace dflash::common
