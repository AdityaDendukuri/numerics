/// @file bench_factorization.cpp
/// @brief 3-way benchmark: our seq vs our omp vs LAPACK for LU and QR.
///
/// For each factorization we register three variants:
///   Backend::seq    -- our implementation, no parallelism
///   Backend::omp    -- our implementation, OpenMP rotation loops
///   Backend::lapack -- LAPACKE_dgetrf / LAPACKE_dgeqrf (industry standard)
///
/// Run with:
///   ./numerics_bench --benchmark_filter=BM_LU
///   ./numerics_bench --benchmark_filter=BM_QR

#include <benchmark/benchmark.h>
#include "numerics.hpp"

using namespace num;

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Generate a diagonally dominant nxn matrix (well-conditioned for LU/QR).
static Matrix make_spd(idx n) {
    Matrix A(n, n, 0.0);
    for (idx i = 0; i < n; ++i) {
        for (idx j = 0; j < n; ++j)
            A(i, j) = static_cast<real>(1 + (i == j ? n : 0))
                      / static_cast<real>(1 + i + j);
        A(i, i) += static_cast<real>(n);  // dominant diagonal
    }
    return A;
}

// ── LU factorization ─────────────────────────────────────────────────────────

template<Backend B>
static void BM_LU(benchmark::State& state) {
    idx n = static_cast<idx>(state.range(0));
    Matrix A = make_spd(n);
    for (auto _ : state) {
        auto f = lu(A, B);
        benchmark::DoNotOptimize(f.LU.data());
    }
    // O(2/3 n^3) flops
    state.counters["GFLOP/s"] = benchmark::Counter(
        2.0 / 3.0 * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n),
        benchmark::Counter::kIsIterationInvariantRate,
        benchmark::Counter::kIs1000);
    state.SetComplexityN(static_cast<int64_t>(n));
}
BENCHMARK_TEMPLATE(BM_LU, Backend::seq)   ->RangeMultiplier(2)->Range(64, 1024)->Complexity();
BENCHMARK_TEMPLATE(BM_LU, Backend::omp)   ->RangeMultiplier(2)->Range(64, 1024)->Complexity();
#if defined(NUMERICS_HAS_LAPACK)
BENCHMARK_TEMPLATE(BM_LU, Backend::lapack)->RangeMultiplier(2)->Range(64, 1024)->Complexity();
#endif

// ── QR factorization ─────────────────────────────────────────────────────────

template<Backend B>
static void BM_QR(benchmark::State& state) {
    idx n = static_cast<idx>(state.range(0));
    Matrix A = make_spd(n);
    for (auto _ : state) {
        auto f = qr(A, B);
        benchmark::DoNotOptimize(f.R.data());
    }
    // O(2 m n^2 - 2/3 n^3) flops for square A
    state.counters["GFLOP/s"] = benchmark::Counter(
        4.0 / 3.0 * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n),
        benchmark::Counter::kIsIterationInvariantRate,
        benchmark::Counter::kIs1000);
    state.SetComplexityN(static_cast<int64_t>(n));
}
BENCHMARK_TEMPLATE(BM_QR, Backend::seq)   ->RangeMultiplier(2)->Range(64, 512)->Complexity();
BENCHMARK_TEMPLATE(BM_QR, Backend::omp)   ->RangeMultiplier(2)->Range(64, 512)->Complexity();
#if defined(NUMERICS_HAS_LAPACK)
BENCHMARK_TEMPLATE(BM_QR, Backend::lapack)->RangeMultiplier(2)->Range(64, 512)->Complexity();
#endif
