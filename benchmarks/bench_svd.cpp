/// @file bench_svd.cpp
/// @brief 3-way benchmark: one-sided Jacobi vs randomized SVD vs LAPACK dgesdd.
///
/// Variants:
///   seq::svd    -- our one-sided Jacobi (full SVD, O(mn*min(m,n)) sweeps)
///   randomized  -- svd_truncated via num::accel (top-k only)
///   lapack::svd -- LAPACKE_dgesdd divide-and-conquer (full SVD)
///
/// Run with:
///   ./numerics_bench --benchmark_filter=BM_SVD

#include "linear/svd/svd.hpp"
#include "numerics.hpp"
#include <benchmark/benchmark.h>

using namespace num;

static mat make_rect(idx m, idx n) {
    mat A(m, n, 0.0);
    for (idx i = 0; i < m; ++i)
        for (idx j = 0; j < n; ++j)
            A(i, j) = static_cast<real>(1) / static_cast<real>(1 + i + j);
    return A;
}

// ── Full SVD ─────────────────────────────────────────────────────────────────

static void BM_SVD_Seq(benchmark::State &state) {
    idx n = static_cast<idx>(state.range(0));
    mat A = make_rect(n, n);
    for (auto _ : state) {
        auto r = seq::svd(A, 1e-12, 100);
        benchmark::DoNotOptimize(r.S.data());
    }
    // O(4/3 n^3) flops (economy SVD)
    state.counters["GFLOP/s"] = benchmark::Counter(
        4.0 / 3.0 * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n),
        benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.SetComplexityN(static_cast<int64_t>(n));
}
BENCHMARK(BM_SVD_Seq)->RangeMultiplier(2)->Range(32, 256)->Complexity();

#if defined(NUMERICS_HAS_LAPACK)
static void BM_SVD_Lapack(benchmark::State &state) {
    idx n = static_cast<idx>(state.range(0));
    mat A = make_rect(n, n);
    for (auto _ : state) {
        auto r = lapack::svd(A);
        benchmark::DoNotOptimize(r.S.data());
    }
    state.counters["GFLOP/s"] = benchmark::Counter(
        4.0 / 3.0 * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n),
        benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.SetComplexityN(static_cast<int64_t>(n));
}
BENCHMARK(BM_SVD_Lapack)->RangeMultiplier(2)->Range(32, 512)->Complexity();
#endif

// ── Randomized truncated SVD
// ────────────────────────────────────────────────── Different algorithm:
// targets top-k only, O(mnk). Not directly comparable for the same n, but shown
// alongside for practical guidance.

static void BM_SVD_Randomized(benchmark::State &state) {
    idx n = static_cast<idx>(state.range(0));
    idx k = std::max(idx(1), n / 8); // top 12.5% singular values
    mat A = make_rect(n, n);
    for (auto _ : state) {
        auto r = svd_truncated(A, k);
        benchmark::DoNotOptimize(r.S.data());
    }
    state.SetComplexityN(static_cast<int64_t>(n));
}
BENCHMARK(BM_SVD_Randomized)->RangeMultiplier(2)->Range(64, 1024)->Complexity();
