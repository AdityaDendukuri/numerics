/// @file bench_autovec.cpp
/// @brief scalar baseline benchmarks  -- compiled WITHOUT auto-vectorization
///
/// This file is compiled with -fno-tree-vectorize -fno-slp-vectorize, so the
/// compiler emits only scalar instructions however wide the target's registers
/// are. That isolates one variable: what the auto-vectorizer is worth. It is a
/// baseline, not a kernel -- nothing in the library takes this path.
///
/// Together with bench_linalg.cpp it gives a four-step comparison:
///
///   1. BM_Matmul_Scalar / BM_Matmul_Scalar_Blocked   (this file)
///      No SIMD at all. One FP multiply-add per cycle is the ceiling.
///
///   2. BM_Matmul_Kernel                              (bench_linalg.cpp)
///      `num::kernel::gemm` on one thread: the same loops written so the
///      compiler can vectorize them, over a register tile sized to the
///      target's register file and a cache panel sized to L2.
///
///   3. BM_Matmul_Omp                                 (bench_linalg.cpp)
///      That kernel per row tile, across cores.
///
///   4. BM_Matmul_Blas                                (bench_linalg.cpp)
///      OpenBLAS/Accelerate: hand-written assembly, runtime CPU dispatch, and
///      on Apple silicon the AMX coprocessor that portable C++ cannot reach.
///      Maintained externally; the practical production ceiling.
///
/// Run just these:
///   ./build/benchmarks/numerics_bench --benchmark_filter=scalar

#include "container/matrix.hpp"
#include <algorithm>
#include <benchmark/benchmark.h>

using namespace num;

// scalar implementations  -- no vectorisation

/// Naive i-j-k triple loop, purely scalar.
/// Inner k-loop reads B column-wise (stride N)  -- cache-hostile.
static void matmul_scalar(const mat &A, const mat &B, mat &C) {
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
static void matmul_scalar_blocked(const mat &A, const mat &B, mat &C) {
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

static void BM_Matmul_Scalar(benchmark::State &state) {
    idx n = static_cast<idx>(state.range(0));
    mat A(n, n, 1.0), B(n, n, 1.0), C(n, n);
    for (auto _ : state) {
        matmul_scalar(A, B, C);
        benchmark::DoNotOptimize(C.data());
    }
    state.counters["GFLOP/s"] = benchmark::Counter(
        flops(n), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.SetComplexityN(static_cast<int64_t>(n));
}
BENCHMARK(BM_Matmul_Scalar)->RangeMultiplier(2)->Range(64, 512)->Complexity();

static void BM_Matmul_Scalar_Blocked(benchmark::State &state) {
    idx n = static_cast<idx>(state.range(0));
    mat A(n, n, 1.0), B(n, n, 1.0), C(n, n);
    for (auto _ : state) {
        matmul_scalar_blocked(A, B, C);
        benchmark::DoNotOptimize(C.data());
    }
    state.counters["GFLOP/s"] = benchmark::Counter(
        flops(n), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.SetComplexityN(static_cast<int64_t>(n));
}
BENCHMARK(BM_Matmul_Scalar_Blocked)->RangeMultiplier(2)->Range(64, 512)->Complexity();
