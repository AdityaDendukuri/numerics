/// @file bench_linalg.cpp
/// @brief Backend-comparative benchmarks for all linear algebra operations
///
/// Each operation is benchmarked across every backend namespace so results
/// appear side-by-side in the output.  Run with:
///
///   ./build/benchmarks/numerics_bench --benchmark_filter=BM_Matmul
///   ./build/benchmarks/numerics_bench --benchmark_format=json > results.json
///
/// Backends benchmarked:
///   seq      -- naive serial C++ (baseline)
///   blocked  -- cache-blocked; compiler auto-vectorizes
///   simd     -- hand-written AVX2/NEON intrinsics
///   blas     -- cblas_dgemm / cblas_dgemv / cblas_ddot (OpenBLAS / MKL)
///   omp      -- OpenMP parallel
///
/// matmul also includes the intermediate optimisation steps
/// (naive -> blocked -> register-blocked -> SIMD -> blas)
/// to illustrate the progression of techniques.

#include "container/matrix_ops.hpp"
#include "container/vector_ops.hpp"
#include "numerics.hpp"
#include "omp/matrix_ops.hpp"
#include "omp/vector_ops.hpp"
#include <benchmark/benchmark.h>

#if defined(NUMERICS_HAS_BLAS)
#include "blas/matrix_ops.hpp"
#include "blas/vector_ops.hpp"
#endif
#ifdef NUMERICS_HAS_CUDA
#include "cuda/container_ops.hpp"
#include <cuda_runtime.h>
#endif

using namespace num;

// Helpers

/// Bytes processed by an NxN matmul: read A+B, write C
static int64_t matmul_bytes(idx n, int64_t iters) {
    return iters * 3 * static_cast<int64_t>(n * n) * sizeof(real);
}

/// FLOP count for an NxN matmul: 2*N^3  (N^2 dot products of length N)
static double matmul_flops(idx n) {
    return 2.0 * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n);
}

// matmul  -- full backend comparison
//
// `Kernel` is the portable `num::kernel::gemm` on one thread: register-tiled
// and cache-panelled, no intrinsics. It is the floor everything else is
// measured against, and the thing `Omp` runs per row tile.

static void BM_Matmul_Kernel(benchmark::State &state) {
    idx n = static_cast<idx>(state.range(0));
    mat A(n, n, 1.0), B(n, n, 1.0), C(n, n);
    for (auto _ : state) {
        seq::matmul(A, B, C);
        benchmark::DoNotOptimize(C.data());
    }
    state.counters["GFLOP/s"] =
        benchmark::Counter(matmul_flops(n), benchmark::Counter::kIsIterationInvariantRate,
                           benchmark::Counter::kIs1000);
    state.SetComplexityN(static_cast<int64_t>(n));
}
BENCHMARK(BM_Matmul_Kernel)->RangeMultiplier(2)->Range(64, 512)->Complexity();


#define NUMERICS_BENCH_MATMUL(name, call)                                                        \
    static void BM_Matmul_##name(benchmark::State &state) {                                      \
        idx n = static_cast<idx>(state.range(0));                                                \
        mat A(n, n, 1.0), B_mat(n, n, 1.0), C(n, n);                                          \
        for (auto _ : state) {                                                                    \
            call;                                                                                 \
            benchmark::DoNotOptimize(C.data());                                                   \
        }                                                                                          \
        state.counters["GFLOP/s"] =                                                               \
            benchmark::Counter(matmul_flops(n), benchmark::Counter::kIsIterationInvariantRate,     \
                               benchmark::Counter::kIs1000);                                       \
        state.SetComplexityN(static_cast<int64_t>(n));                                            \
    }                                                                                              \
    BENCHMARK(BM_Matmul_##name)->RangeMultiplier(2)->Range(64, 512)->Complexity()

#if defined(NUMERICS_HAS_BLAS)
NUMERICS_BENCH_MATMUL(Blas, blas::matmul(A, B_mat, C));
#endif
NUMERICS_BENCH_MATMUL(Omp, omp::matmul(A, B_mat, C));

#ifdef NUMERICS_HAS_CUDA
static void BM_Matmul_GPU(benchmark::State &state) {
    idx n = static_cast<idx>(state.range(0));
    mat A(n, n, 1.0), B(n, n, 1.0), C(n, n);
    A.to_gpu();
    B.to_gpu();
    C.to_gpu();
    cudaDeviceSynchronize();
    for (auto _ : state) {
        cuda::matmul(A, B, C);
        cudaDeviceSynchronize();
        benchmark::DoNotOptimize(C.gpu_data());
    }
    state.counters["GFLOP/s"] =
        benchmark::Counter(matmul_flops(n), benchmark::Counter::kIsIterationInvariantRate,
                           benchmark::Counter::kIs1000);
    state.SetComplexityN(static_cast<int64_t>(n));
}
BENCHMARK(BM_Matmul_GPU)->RangeMultiplier(2)->Range(64, 512)->Complexity();
#endif

// matvec  -- backend comparison

#define NUMERICS_BENCH_MATVEC(name, call)                                                        \
    static void BM_Matvec_##name(benchmark::State &state) {                                      \
        idx n = static_cast<idx>(state.range(0));                                                \
        mat A(n, n, 1.0);                                                                      \
        vec x(n, 1.0), y(n);                                                                   \
        for (auto _ : state) {                                                                    \
            call;                                                                                 \
            benchmark::DoNotOptimize(y.data());                                                   \
        }                                                                                          \
        state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(n * n + 2 * n) *         \
                                sizeof(real));                                                     \
    }                                                                                              \
    BENCHMARK(BM_Matvec_##name)->RangeMultiplier(2)->Range(64, 2048)

NUMERICS_BENCH_MATVEC(Seq, seq::matvec(A, x, y));
NUMERICS_BENCH_MATVEC(Omp, omp::matvec(A, x, y));
#if defined(NUMERICS_HAS_BLAS)
NUMERICS_BENCH_MATVEC(Blas, blas::matvec(A, x, y));
#endif

#ifdef NUMERICS_HAS_CUDA
static void BM_Matvec_GPU(benchmark::State &state) {
    idx n = static_cast<idx>(state.range(0));
    mat A(n, n, 1.0);
    vec x(n, 1.0), y(n);
    A.to_gpu();
    x.to_gpu();
    y.to_gpu();
    cudaDeviceSynchronize();
    for (auto _ : state) {
        cuda::matvec(A, x, y);
        cudaDeviceSynchronize();
        benchmark::DoNotOptimize(y.gpu_data());
    }
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(n * n + 2 * n) *
                            sizeof(real));
}
BENCHMARK(BM_Matvec_GPU)->RangeMultiplier(2)->Range(64, 2048);
#endif

// dot product  -- backend comparison

#define NUMERICS_BENCH_DOT(name, call)                                                           \
    static void BM_Dot_##name(benchmark::State &state) {                                         \
        vec x(static_cast<idx>(state.range(0)), 1.0);                                         \
        vec y(static_cast<idx>(state.range(0)), 2.0);                                         \
        for (auto _ : state) {                                                                    \
            benchmark::DoNotOptimize(call);                                                       \
        }                                                                                          \
        state.SetBytesProcessed(state.iterations() * state.range(0) * 2 * sizeof(real));           \
    }                                                                                              \
    BENCHMARK(BM_Dot_##name)->RangeMultiplier(4)->Range(1024, 1 << 20)

NUMERICS_BENCH_DOT(Seq, seq::dot(x, y));
NUMERICS_BENCH_DOT(Omp, omp::dot(x, y));
#if defined(NUMERICS_HAS_BLAS)
NUMERICS_BENCH_DOT(Blas, blas::dot(x, y));
#endif

#ifdef NUMERICS_HAS_CUDA
static void BM_Dot_GPU(benchmark::State &state) {
    vec x(static_cast<idx>(state.range(0)), 1.0);
    vec y(static_cast<idx>(state.range(0)), 2.0);
    x.to_gpu();
    y.to_gpu();
    cudaDeviceSynchronize();
    for (auto _ : state) {
        benchmark::DoNotOptimize(cuda::dot(x, y));
        cudaDeviceSynchronize();
    }
    state.SetBytesProcessed(state.iterations() * state.range(0) * 2 * sizeof(real));
}
BENCHMARK(BM_Dot_GPU)->RangeMultiplier(4)->Range(1024, 1 << 20);
#endif

// axpy  -- backend comparison

#define NUMERICS_BENCH_AXPY(name, call)                                                          \
    static void BM_Axpy_##name(benchmark::State &state) {                                        \
        vec x(static_cast<idx>(state.range(0)), 1.0);                                         \
        vec y(static_cast<idx>(state.range(0)), 2.0);                                         \
        for (auto _ : state) {                                                                    \
            call;                                                                                 \
            benchmark::DoNotOptimize(y.data());                                                   \
        }                                                                                          \
        state.SetBytesProcessed(state.iterations() * state.range(0) * 3 * sizeof(real));           \
    }                                                                                              \
    BENCHMARK(BM_Axpy_##name)->RangeMultiplier(4)->Range(1024, 1 << 20)

NUMERICS_BENCH_AXPY(Seq, seq::axpy(2.0, x, y));
NUMERICS_BENCH_AXPY(Omp, omp::axpy(2.0, x, y));
#if defined(NUMERICS_HAS_BLAS)
NUMERICS_BENCH_AXPY(Blas, blas::axpy(2.0, x, y));
#endif

#ifdef NUMERICS_HAS_CUDA
static void BM_Axpy_GPU(benchmark::State &state) {
    vec x(static_cast<idx>(state.range(0)), 1.0);
    vec y(static_cast<idx>(state.range(0)), 2.0);
    x.to_gpu();
    y.to_gpu();
    cudaDeviceSynchronize();
    for (auto _ : state) {
        cuda::axpy(2.0, x, y);
        cudaDeviceSynchronize();
        benchmark::DoNotOptimize(y.gpu_data());
    }
    state.SetBytesProcessed(state.iterations() * state.range(0) * 3 * sizeof(real));
}
BENCHMARK(BM_Axpy_GPU)->RangeMultiplier(4)->Range(1024, 1 << 20);
#endif

// Conjugate Gradient solver

static void BM_CG(benchmark::State &state) {
    idx n = static_cast<idx>(state.range(0));
    mat A(n, n, 0.0);
    for (idx i = 0; i < n; ++i) {
        A(i, i) = static_cast<real>(n);
        if (i > 0)
            A(i, i - 1) = 1.0;
        if (i < n - 1)
            A(i, i + 1) = 1.0;
    }
    vec b(n, 1.0), x(n, 0.0);
    for (auto _ : state) {
        state.PauseTiming();
        for (idx i = 0; i < n; ++i)
            x[i] = 0.0;
        state.ResumeTiming();
        auto r = cg(assume_spd(A), b, x);
        benchmark::DoNotOptimize(x.data());
        state.counters["iters"] = static_cast<double>(r.iterations);
    }
    state.SetComplexityN(static_cast<int64_t>(n));
}
BENCHMARK(BM_CG)->RangeMultiplier(2)->Range(32, 256)->Complexity();

#ifdef NUMERICS_HAS_CUDA
static void BM_CG_GPU(benchmark::State &state) {
    idx n = static_cast<idx>(state.range(0));
    mat A(n, n, 0.0);
    for (idx i = 0; i < n; ++i) {
        A(i, i) = static_cast<real>(n);
        if (i > 0)
            A(i, i - 1) = 1.0;
        if (i < n - 1)
            A(i, i + 1) = 1.0;
    }
    vec b(n, 1.0), x(n, 0.0);
    for (auto _ : state) {
        state.PauseTiming();
        for (idx i = 0; i < n; ++i)
            x[i] = 0.0;
        state.ResumeTiming();
        auto r = unsafe::cuda::cg(A, b, x, 1e-10, 1000);
        cudaDeviceSynchronize();
        benchmark::DoNotOptimize(x.data());
        state.counters["iters"] = static_cast<double>(r.iterations);
    }
    state.SetComplexityN(static_cast<int64_t>(n));
}
BENCHMARK(BM_CG_GPU)->RangeMultiplier(2)->Range(32, 256)->Complexity();
#endif

// Thomas algorithm (tridiagonal solver)  -- no parallel equivalent

static void BM_Thomas(benchmark::State &state) {
    idx n = static_cast<idx>(state.range(0));
    vec a(n - 1, -1.0), b(n, 2.0), c(n - 1, -1.0), d(n, 1.0), x(n);
    for (auto _ : state) {
        thomas(a, b, c, d, x);
        benchmark::DoNotOptimize(x.data());
    }
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(5 * n - 2) * sizeof(real));
    state.SetComplexityN(static_cast<int64_t>(n));
}
BENCHMARK(BM_Thomas)->RangeMultiplier(4)->Range(64, 1 << 16)->Complexity();
