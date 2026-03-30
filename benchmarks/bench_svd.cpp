/// @file bench_svd.cpp
/// @brief 3-way benchmark: one-sided Jacobi vs randomized SVD vs LAPACK dgesdd.
///
/// Variants:
///   Backend::seq    -- our one-sided Jacobi (full SVD, O(mn*min(m,n)) sweeps)
///   randomized      -- svd_truncated with Backend::blas sketch (top-k only)
///   Backend::lapack -- LAPACKE_dgesdd divide-and-conquer (full SVD)
///
/// Run with:
///   ./numerics_bench --benchmark_filter=BM_SVD

#include <benchmark/benchmark.h>
#include "numerics.hpp"
#include "linalg/svd/svd.hpp"

using namespace num;

static Matrix make_rect(idx m, idx n) {
    Matrix A(m, n, 0.0);
    for (idx i = 0; i < m; ++i)
        for (idx j = 0; j < n; ++j)
            A(i, j) = static_cast<real>(1) / static_cast<real>(1 + i + j);
    return A;
}

// ── Full SVD ─────────────────────────────────────────────────────────────────

template<Backend B>
static void BM_SVD(benchmark::State& state) {
    idx n = static_cast<idx>(state.range(0));
    Matrix A = make_rect(n, n);
    for (auto _ : state) {
        auto r = svd(A, B);
        benchmark::DoNotOptimize(r.S.data());
    }
    // O(4/3 n^3) flops (economy SVD)
    state.counters["GFLOP/s"] = benchmark::Counter(
        4.0 / 3.0 * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n),
        benchmark::Counter::kIsIterationInvariantRate,
        benchmark::Counter::kIs1000);
    state.SetComplexityN(static_cast<int64_t>(n));
}
BENCHMARK_TEMPLATE(BM_SVD, Backend::seq)   ->RangeMultiplier(2)->Range(32, 256)->Complexity();
#if defined(NUMERICS_HAS_LAPACK)
BENCHMARK_TEMPLATE(BM_SVD, Backend::lapack)->RangeMultiplier(2)->Range(32, 512)->Complexity();
#endif

// ── Randomized truncated SVD ──────────────────────────────────────────────────
// Different algorithm: targets top-k only, O(mnk). Not directly comparable
// for the same n, but shown alongside for practical guidance.

static void BM_SVD_Randomized(benchmark::State& state) {
    idx n  = static_cast<idx>(state.range(0));
    idx k  = std::max(idx(1), n / 8);   // top 12.5% singular values
    Matrix A = make_rect(n, n);
    for (auto _ : state) {
        auto r = svd_truncated(A, k, default_backend);
        benchmark::DoNotOptimize(r.S.data());
    }
    state.SetComplexityN(static_cast<int64_t>(n));
}
BENCHMARK(BM_SVD_Randomized)->RangeMultiplier(2)->Range(64, 1024)->Complexity();
