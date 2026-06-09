// Unit tests for dflash::common::verify_derived_scalars — no GPU, no model files.
//
// Build:  cmake --build build --target test_derived_scalars -j
// Run:    cd build && ctest -R derived_scalars --output-on-failure

#include "common/derived_scalars.h"

#include <cstdio>
#include <string>

using namespace dflash::common;

// ─── Minimal test framework ────────────────────────────────────────────────────

static int test_failures = 0;
static int test_count    = 0;

#define TEST_ASSERT(expr) do { \
    test_count++; \
    if (!(expr)) { \
        test_failures++; \
        std::fprintf(stderr, "  FAIL: %s:%d: %s\n", __FILE__, __LINE__, #expr); \
    } \
} while (0)

#define RUN_TEST(fn) do { \
    std::fprintf(stderr, "  %s ...", #fn); \
    int before = test_failures; \
    fn(); \
    if (test_failures == before) std::fprintf(stderr, " ok\n"); \
    else std::fprintf(stderr, "\n"); \
} while (0)

// ─── Tests ─────────────────────────────────────────────────────────────────────

// All three dims match: returns true, err untouched.
static void match_returns_true() {
    std::string err;
    bool ok = verify_derived_scalars(
        /*wq_ne1*/ 4096, /*wk_ne1*/ 512, /*wq_ne0*/ 5120,
        /*exp_q_dim*/ 4096, /*exp_kv_dim*/ 512, /*exp_n_embd*/ 5120,
        "blk.0", err);
    TEST_ASSERT(ok);
    TEST_ASSERT(err.empty());
}

// wq_ne1 != expected_q_dim: returns false and err non-empty.
static void mismatch_q_dim_returns_false() {
    std::string err;
    bool ok = verify_derived_scalars(
        /*wq_ne1*/ 4096 + 1, /*wk_ne1*/ 512, /*wq_ne0*/ 5120,
        /*exp_q_dim*/ 4096, /*exp_kv_dim*/ 512, /*exp_n_embd*/ 5120,
        "blk.0", err);
    TEST_ASSERT(!ok);
    TEST_ASSERT(!err.empty());
}

// wk_ne1 != expected_kv_dim: returns false.
static void mismatch_kv_dim_returns_false() {
    std::string err;
    bool ok = verify_derived_scalars(
        /*wq_ne1*/ 4096, /*wk_ne1*/ 512 + 1, /*wq_ne0*/ 5120,
        /*exp_q_dim*/ 4096, /*exp_kv_dim*/ 512, /*exp_n_embd*/ 5120,
        "blk.0", err);
    TEST_ASSERT(!ok);
    TEST_ASSERT(!err.empty());
}

// wq_ne0 != expected_n_embd: returns false.
static void mismatch_n_embd_returns_false() {
    std::string err;
    bool ok = verify_derived_scalars(
        /*wq_ne1*/ 4096, /*wk_ne1*/ 512, /*wq_ne0*/ 5120 + 64,
        /*exp_q_dim*/ 4096, /*exp_kv_dim*/ 512, /*exp_n_embd*/ 5120,
        "blk.0", err);
    TEST_ASSERT(!ok);
    TEST_ASSERT(!err.empty());
}

// Typical draft model dims: n_head=32, head_dim=128, n_head_kv=8, n_embd=5120.
// expected_q_dim=32*128=4096, expected_kv_dim=8*128=1024.
static void draft_dims_match() {
    std::string err;
    bool ok = verify_derived_scalars(
        4096, 1024, 5120,
        (int64_t)32 * 128, (int64_t)8 * 128, 5120,
        "blk.0", err);
    TEST_ASSERT(ok);
    TEST_ASSERT(err.empty());
}

// Typical qwen35 target layer: n_head=24, n_embd_head_k=256, n_head_kv=4.
// expected_q_dim = 24*256*2 = 12288 (Q+gate packed).
// expected_kv_dim = 4*256 = 1024.
static void qwen35_target_dims_match() {
    std::string err;
    bool ok = verify_derived_scalars(
        /*wq_ne1*/ 12288, /*wk_ne1*/ 1024, /*wq_ne0*/ 5120,
        /*exp_q_dim*/ (int64_t)24 * 256 * 2,
        /*exp_kv_dim*/ (int64_t)4 * 256,
        /*exp_n_embd*/ 5120,
        "blk.3", err);
    TEST_ASSERT(ok);
    TEST_ASSERT(err.empty());
}

// Error message must contain the layer tag for easy diagnosis.
static void err_contains_layer_tag() {
    std::string err;
    verify_derived_scalars(
        4097, 1024, 5120,
        4096, 1024, 5120,
        "blk.15", err);
    TEST_ASSERT(err.find("blk.15") != std::string::npos);
}

// ─── main ──────────────────────────────────────────────────────────────────────

int main() {
    std::fprintf(stderr, "=== test_derived_scalars ===\n");

    RUN_TEST(match_returns_true);
    RUN_TEST(mismatch_q_dim_returns_false);
    RUN_TEST(mismatch_kv_dim_returns_false);
    RUN_TEST(mismatch_n_embd_returns_false);
    RUN_TEST(draft_dims_match);
    RUN_TEST(qwen35_target_dims_match);
    RUN_TEST(err_contains_layer_tag);

    std::fprintf(stderr, "\n%d tests, %d failures\n", test_count, test_failures);
    return (test_failures == 0) ? 0 : 1;
}
