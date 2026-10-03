/// @file bench_factorization.cpp
/// @brief 2-way benchmark: our seq vs LAPACK for LU and QR.
///
/// For each factorization we register two variants:
///   seq::lu / seq::qr       -- our implementation, no parallelism
///   lapack::lu / lapack::qr -- LAPACKE_dgetrf / LAPACKE_dgeqrf (industry standard)
///
/// There is no OMP variant: neither `lu` nor `qr` has a distinct OpenMP path
/// (see kernel/factor.hpp), so it would just re-measure the seq code under a
/// different name.
///
/// Run with:
///   ./numerics_bench --benchmark_filter=BM_LU
///   ./numerics_bench --benchmark_filter=BM_QR

#include "numerics.hpp"
#include <benchmark/benchmark.h>

using namespace num;

// Helpers

/// Generate a diagonally dominant nxn matrix (well-conditioned for LU/QR).
static mat<real> make_spd(idx n) {
    mat<real> A(n, n, 0.0);
    for (idx i = 0; i < n; ++i) {
        for (idx j = 0; j < n; ++j)
            A(i, j) = static_cast<real>(1 + (i == j ? n : 0)) / static_cast<real>(1 + i + j);
        A(i, i) += static_cast<real>(n); // dominant diagonal
    }
    return A;
}

// LU factorization

static void BM_LU_Seq(benchmark::State &state) {
    idx n = static_cast<idx>(state.range(0));
    mat<real> A = make_spd(n);
    for (auto _ : state) {
        auto f = seq::lu(A);
        benchmark::DoNotOptimize(f.LU.data());
    }
    // O(2/3 n^3) flops
    state.counters["GFLOP/s"] = benchmark::Counter(
        2.0 / 3.0 * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n),
        benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.SetComplexityN(static_cast<int64_t>(n));
}
BENCHMARK(BM_LU_Seq)->RangeMultiplier(2)->Range(64, 1024)->Complexity();

// The same kernel in single precision: twice the values per SIMD register, half the bytes.
static void BM_LU_Seq_Float(benchmark::State &state) {
    idx n = static_cast<idx>(state.range(0));
    const mat<real> A = make_spd(n);
    mat<float> A32(n, n);
    for (idx k = 0; k < n * n; ++k) {
        A32.data()[k] = static_cast<float>(A.data()[k]);
    }
    for (auto _ : state) {
        auto f = seq::lu(A32);
        benchmark::DoNotOptimize(f.LU.data());
    }
    state.counters["GFLOP/s"] = benchmark::Counter(
        2.0 / 3.0 * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n),
        benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}
BENCHMARK(BM_LU_Seq_Float)->RangeMultiplier(2)->Range(128, 2048);

// Factor and solve one right-hand side to double accuracy: double LU against float LU with
// iterative refinement, both through the in-tree kernel.
static void BM_Solve_DoubleLU(benchmark::State &state) {
    idx n = static_cast<idx>(state.range(0));
    const mat<real> A = make_spd(n);
    const vec<real> b(n, 1.0);
    vec<real> x;
    for (auto _ : state) {
        solve(seq::lu(A), b, x);
        benchmark::DoNotOptimize(x.data());
    }
}
BENCHMARK(BM_Solve_DoubleLU)->RangeMultiplier(2)->Range(128, 2048)->Unit(benchmark::kMillisecond);

static void BM_Solve_MixedLU(benchmark::State &state) {
    idx n = static_cast<idx>(state.range(0));
    const mat<real> A = make_spd(n);
    const vec<real> b(n, 1.0);
    vec<real> x;
    for (auto _ : state) {
        solve(lu(A, mixed_precision), b, x);
        benchmark::DoNotOptimize(x.data());
    }
}
BENCHMARK(BM_Solve_MixedLU)->RangeMultiplier(2)->Range(128, 2048)->Unit(benchmark::kMillisecond);

#if defined(NUMERICS_HAS_LAPACK)
static void BM_LU_Lapack(benchmark::State &state) {
    idx n = static_cast<idx>(state.range(0));
    mat<real> A = make_spd(n);
    for (auto _ : state) {
        auto f = lapack::lu(A);
        benchmark::DoNotOptimize(f.LU.data());
    }
    state.counters["GFLOP/s"] = benchmark::Counter(
        2.0 / 3.0 * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n),
        benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.SetComplexityN(static_cast<int64_t>(n));
}
BENCHMARK(BM_LU_Lapack)->RangeMultiplier(2)->Range(64, 1024)->Complexity();
#endif

// QR factorization

static void BM_QR_Seq(benchmark::State &state) {
    idx n = static_cast<idx>(state.range(0));
    mat<real> A = make_spd(n);
    for (auto _ : state) {
        auto f = seq::qr(A);
        benchmark::DoNotOptimize(f.R.data());
    }
    // O(2 m n^2 - 2/3 n^3) flops for square A
    state.counters["GFLOP/s"] = benchmark::Counter(
        4.0 / 3.0 * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n),
        benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.SetComplexityN(static_cast<int64_t>(n));
}
BENCHMARK(BM_QR_Seq)->RangeMultiplier(2)->Range(64, 512)->Complexity();

#if defined(NUMERICS_HAS_LAPACK)
static void BM_QR_Lapack(benchmark::State &state) {
    idx n = static_cast<idx>(state.range(0));
    mat<real> A = make_spd(n);
    for (auto _ : state) {
        auto f = lapack::qr(A);
        benchmark::DoNotOptimize(f.R.data());
    }
    state.counters["GFLOP/s"] = benchmark::Counter(
        4.0 / 3.0 * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n),
        benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.SetComplexityN(static_cast<int64_t>(n));
}
BENCHMARK(BM_QR_Lapack)->RangeMultiplier(2)->Range(64, 512)->Complexity();
#endif
