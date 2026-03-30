/// @file bench_autovec.cpp
/// @brief Scalar baseline benchmarks  -- compiled WITHOUT auto-vectorization
///
/// This file is compiled with -fno-tree-vectorize -fno-slp-vectorize so the
/// compiler emits only scalar instructions, even though -mavx2 is active.
/// The distinction matters:
///
///   -mavx2               -> "AVX2 instructions exist on this CPU"
///   -fno-tree-vectorize  -> "don't use the auto-vectoriser to emit them"
///
/// Combined with bench_linalg.cpp this gives a three-tier comparison:
///
///   Tier 1  BM_Matmul_Scalar / BM_Matmul_Scalar_Blocked  (this file)
///           No SIMD at all.  Pure scalar throughput.
///           Bottleneck: one FP multiply-add per cycle.
///
///   Tier 2  BM_Matmul_Naive / BM_Matmul_Blocked          (bench_linalg.cpp)
///           Compiled with -mavx2.  Compiler auto-vectorises what it can.
///           Shows how much the compiler can do on its own with good loop order.
///
///   Tier 3  BM_Matmul<Backend::simd>                      (bench_linalg.cpp)
///           Explicit AVX2 intrinsics + cache tiling written by hand.
///           Shows the ceiling of CPU matmul without a tuned BLAS.
///
///   Tier 4  BM_Matmul<Backend::blas>                      (bench_linalg.cpp)
///           OpenBLAS: hand-written assembly kernels, runtime CPU dispatch.
///           Maintained externally; the practical production ceiling.
///
/// Run just these:
///   ./build/benchmarks/numerics_bench --benchmark_filter=Scalar

#include <benchmark/benchmark.h>
#include "core/matrix.hpp"
#include <algorithm>

using namespace num;

// Scalar implementations  -- no vectorisation

/// Naive i-j-k triple loop, purely scalar.
/// Inner k-loop reads B column-wise (stride N)  -- cache-hostile.
static void matmul_scalar(const Matrix& A, const Matrix& B, Matrix& C) {
    const idx M = A.rows(), K = A.cols(), N = B.cols();
    for (idx i = 0; i < M; ++i)
        for (idx j = 0; j < N; ++j) {
            real sum = 0;
            for (idx k = 0; k < K; ++k)
                sum += A(i, k) * B(k, j);
            C(i, j) = sum;
        }
}

/// Cache-blocked i-k-j with 64-wide tiles, purely scalar.
/// Shows that cache efficiency alone gives a large speedup even without SIMD.
static void matmul_scalar_blocked(const Matrix& A, const Matrix& B, Matrix& C) {
    constexpr idx BS = 64;
    const idx M = A.rows(), K = A.cols(), N = B.cols();
    std::fill_n(C.data(), M * N, real(0));

    for (idx ii = 0; ii < M; ii += BS) {
        const idx i_lim = std::min(ii + BS, M);
        for (idx jj = 0; jj < N; jj += BS) {
            const idx j_lim = std::min(jj + BS, N);
            for (idx kk = 0; kk < K; kk += BS) {
                const idx k_lim = std::min(kk + BS, K);
                for (idx i = ii; i < i_lim; ++i) {
                    for (idx k = kk; k < k_lim; ++k) {
                        const real a_ik = A(i, k);
                        for (idx j = jj; j < j_lim; ++j)
                            C(i, j) += a_ik * B(k, j);
                    }
                }
            }
        }
    }
}

// Benchmarks

static double flops(idx n) {
    return 2.0 * double(n) * double(n) * double(n);
}

static void BM_Matmul_Scalar(benchmark::State& state) {
    idx n = static_cast<idx>(state.range(0));
    Matrix A(n, n, 1.0), B(n, n, 1.0), C(n, n);
    for (auto _ : state) {
        matmul_scalar(A, B, C);
        benchmark::DoNotOptimize(C.data());
    }
    state.counters["GFLOP/s"] = benchmark::Counter(
        flops(n), benchmark::Counter::kIsIterationInvariantRate,
        benchmark::Counter::kIs1000);
    state.SetComplexityN(static_cast<int64_t>(n));
}
BENCHMARK(BM_Matmul_Scalar)->RangeMultiplier(2)->Range(64, 512)->Complexity();

static void BM_Matmul_Scalar_Blocked(benchmark::State& state) {
    idx n = static_cast<idx>(state.range(0));
    Matrix A(n, n, 1.0), B(n, n, 1.0), C(n, n);
    for (auto _ : state) {
        matmul_scalar_blocked(A, B, C);
        benchmark::DoNotOptimize(C.data());
    }
    state.counters["GFLOP/s"] = benchmark::Counter(
        flops(n), benchmark::Counter::kIsIterationInvariantRate,
        benchmark::Counter::kIs1000);
    state.SetComplexityN(static_cast<int64_t>(n));
}
BENCHMARK(BM_Matmul_Scalar_Blocked)->RangeMultiplier(2)->Range(64, 512)->Complexity();
