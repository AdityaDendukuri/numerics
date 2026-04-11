/// @file bench_linalg.cpp
/// @brief Backend-comparative benchmarks for all linear algebra operations
///
/// Each operation is benchmarked across every execution backend so results
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

#include <benchmark/benchmark.h>
#include "numerics.hpp"

#ifdef NUMERICS_HAS_CUDA
#  include <cuda_runtime.h>
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

// --- Intermediate optimisation steps (seq -> blocked -> reg-blocked) ----------
// These three show the manual progression before SIMD/BLAS.

static void BM_Matmul_Naive(benchmark::State& state) {
    idx n = static_cast<idx>(state.range(0));
    Matrix A(n, n, 1.0), B(n, n, 1.0), C(n, n);
    for (auto _ : state) {
        matmul(A, B, C, seq);
        benchmark::DoNotOptimize(C.data());
    }
    state.counters["GFLOP/s"] = benchmark::Counter(
        matmul_flops(n), benchmark::Counter::kIsIterationInvariantRate,
        benchmark::Counter::kIs1000);
    state.SetComplexityN(static_cast<int64_t>(n));
}
BENCHMARK(BM_Matmul_Naive)->RangeMultiplier(2)->Range(64, 512)->Complexity();

static void BM_Matmul_Blocked(benchmark::State& state) {
    idx n = static_cast<idx>(state.range(0));
    Matrix A(n, n, 1.0), B(n, n, 1.0), C(n, n);
    for (auto _ : state) {
        matmul_blocked(A, B, C);
        benchmark::DoNotOptimize(C.data());
    }
    state.counters["GFLOP/s"] = benchmark::Counter(
        matmul_flops(n), benchmark::Counter::kIsIterationInvariantRate,
        benchmark::Counter::kIs1000);
    state.SetComplexityN(static_cast<int64_t>(n));
}
BENCHMARK(BM_Matmul_Blocked)->RangeMultiplier(2)->Range(64, 512)->Complexity();

static void BM_Matmul_RegBlocked(benchmark::State& state) {
    idx n = static_cast<idx>(state.range(0));
    Matrix A(n, n, 1.0), B(n, n, 1.0), C(n, n);
    for (auto _ : state) {
        matmul_register_blocked(A, B, C);
        benchmark::DoNotOptimize(C.data());
    }
    state.counters["GFLOP/s"] = benchmark::Counter(
        matmul_flops(n), benchmark::Counter::kIsIterationInvariantRate,
        benchmark::Counter::kIs1000);
    state.SetComplexityN(static_cast<int64_t>(n));
}
BENCHMARK(BM_Matmul_RegBlocked)->RangeMultiplier(2)->Range(64, 512)->Complexity();


template<Backend B>
static void BM_Matmul(benchmark::State& state) {
    idx n = static_cast<idx>(state.range(0));
    Matrix A(n, n, 1.0), B_mat(n, n, 1.0), C(n, n);
    for (auto _ : state) {
        matmul(A, B_mat, C, B);
        benchmark::DoNotOptimize(C.data());
    }
    state.counters["GFLOP/s"] = benchmark::Counter(
        matmul_flops(n), benchmark::Counter::kIsIterationInvariantRate,
        benchmark::Counter::kIs1000);
    state.SetComplexityN(static_cast<int64_t>(n));
}
BENCHMARK_TEMPLATE(BM_Matmul, Backend::blocked)->RangeMultiplier(2)->Range(64, 512)->Complexity();
BENCHMARK_TEMPLATE(BM_Matmul, Backend::simd)   ->RangeMultiplier(2)->Range(64, 512)->Complexity();
BENCHMARK_TEMPLATE(BM_Matmul, Backend::blas)   ->RangeMultiplier(2)->Range(64, 512)->Complexity();
BENCHMARK_TEMPLATE(BM_Matmul, Backend::omp)    ->RangeMultiplier(2)->Range(64, 512)->Complexity();

#ifdef NUMERICS_HAS_CUDA
static void BM_Matmul_GPU(benchmark::State& state) {
    idx n = static_cast<idx>(state.range(0));
    Matrix A(n, n, 1.0), B(n, n, 1.0), C(n, n);
    A.to_gpu(); B.to_gpu(); C.to_gpu();
    cudaDeviceSynchronize();
    for (auto _ : state) {
        matmul(A, B, C, gpu);
        cudaDeviceSynchronize();
        benchmark::DoNotOptimize(C.gpu_data());
    }
    state.counters["GFLOP/s"] = benchmark::Counter(
        matmul_flops(n), benchmark::Counter::kIsIterationInvariantRate,
        benchmark::Counter::kIs1000);
    state.SetComplexityN(static_cast<int64_t>(n));
}
BENCHMARK(BM_Matmul_GPU)->RangeMultiplier(2)->Range(64, 512)->Complexity();
#endif

// matvec  -- backend comparison

template<Backend B>
static void BM_Matvec(benchmark::State& state) {
    idx n = static_cast<idx>(state.range(0));
    Matrix A(n, n, 1.0);
    Vector x(n, 1.0), y(n);
    for (auto _ : state) {
        matvec(A, x, y, B);
        benchmark::DoNotOptimize(y.data());
    }
    // reads n^2 + n doubles, writes n doubles
    state.SetBytesProcessed(
        state.iterations() * static_cast<int64_t>(n * n + 2 * n) * sizeof(real));
}
BENCHMARK_TEMPLATE(BM_Matvec, Backend::seq)    ->RangeMultiplier(2)->Range(64, 2048);
BENCHMARK_TEMPLATE(BM_Matvec, Backend::blocked)->RangeMultiplier(2)->Range(64, 2048);
BENCHMARK_TEMPLATE(BM_Matvec, Backend::simd)   ->RangeMultiplier(2)->Range(64, 2048);
BENCHMARK_TEMPLATE(BM_Matvec, Backend::blas)   ->RangeMultiplier(2)->Range(64, 2048);
BENCHMARK_TEMPLATE(BM_Matvec, Backend::omp)    ->RangeMultiplier(2)->Range(64, 2048);

#ifdef NUMERICS_HAS_CUDA
static void BM_Matvec_GPU(benchmark::State& state) {
    idx n = static_cast<idx>(state.range(0));
    Matrix A(n, n, 1.0);
    Vector x(n, 1.0), y(n);
    A.to_gpu(); x.to_gpu(); y.to_gpu();
    cudaDeviceSynchronize();
    for (auto _ : state) {
        matvec(A, x, y, gpu);
        cudaDeviceSynchronize();
        benchmark::DoNotOptimize(y.gpu_data());
    }
    state.SetBytesProcessed(
        state.iterations() * static_cast<int64_t>(n * n + 2 * n) * sizeof(real));
}
BENCHMARK(BM_Matvec_GPU)->RangeMultiplier(2)->Range(64, 2048);
#endif

// dot product  -- backend comparison

template<Backend B>
static void BM_Dot(benchmark::State& state) {
    Vector x(static_cast<idx>(state.range(0)), 1.0);
    Vector y(static_cast<idx>(state.range(0)), 2.0);
    for (auto _ : state) {
        benchmark::DoNotOptimize(dot(x, y, B));
    }
    // reads 2n doubles
    state.SetBytesProcessed(
        state.iterations() * state.range(0) * 2 * sizeof(real));
}
BENCHMARK_TEMPLATE(BM_Dot, Backend::seq) ->RangeMultiplier(4)->Range(1024, 1 << 20);
BENCHMARK_TEMPLATE(BM_Dot, Backend::blas)->RangeMultiplier(4)->Range(1024, 1 << 20);
BENCHMARK_TEMPLATE(BM_Dot, Backend::omp) ->RangeMultiplier(4)->Range(1024, 1 << 20);

#ifdef NUMERICS_HAS_CUDA
static void BM_Dot_GPU(benchmark::State& state) {
    Vector x(static_cast<idx>(state.range(0)), 1.0);
    Vector y(static_cast<idx>(state.range(0)), 2.0);
    x.to_gpu(); y.to_gpu();
    cudaDeviceSynchronize();
    for (auto _ : state) {
        benchmark::DoNotOptimize(dot(x, y, gpu));
        cudaDeviceSynchronize();
    }
    state.SetBytesProcessed(
        state.iterations() * state.range(0) * 2 * sizeof(real));
}
BENCHMARK(BM_Dot_GPU)->RangeMultiplier(4)->Range(1024, 1 << 20);
#endif

// axpy  -- backend comparison

template<Backend B>
static void BM_Axpy(benchmark::State& state) {
    Vector x(static_cast<idx>(state.range(0)), 1.0);
    Vector y(static_cast<idx>(state.range(0)), 2.0);
    for (auto _ : state) {
        axpy(2.0, x, y, B);
        benchmark::DoNotOptimize(y.data());
    }
    // reads 2n, writes n doubles
    state.SetBytesProcessed(
        state.iterations() * state.range(0) * 3 * sizeof(real));
}
BENCHMARK_TEMPLATE(BM_Axpy, Backend::seq) ->RangeMultiplier(4)->Range(1024, 1 << 20);
BENCHMARK_TEMPLATE(BM_Axpy, Backend::blas)->RangeMultiplier(4)->Range(1024, 1 << 20);
BENCHMARK_TEMPLATE(BM_Axpy, Backend::omp) ->RangeMultiplier(4)->Range(1024, 1 << 20);

#ifdef NUMERICS_HAS_CUDA
static void BM_Axpy_GPU(benchmark::State& state) {
    Vector x(static_cast<idx>(state.range(0)), 1.0);
    Vector y(static_cast<idx>(state.range(0)), 2.0);
    x.to_gpu(); y.to_gpu();
    cudaDeviceSynchronize();
    for (auto _ : state) {
        axpy(2.0, x, y, gpu);
        cudaDeviceSynchronize();
        benchmark::DoNotOptimize(y.gpu_data());
    }
    state.SetBytesProcessed(
        state.iterations() * state.range(0) * 3 * sizeof(real));
}
BENCHMARK(BM_Axpy_GPU)->RangeMultiplier(4)->Range(1024, 1 << 20);
#endif

// Conjugate Gradient solver

static void BM_CG(benchmark::State& state) {
    idx n = static_cast<idx>(state.range(0));
    Matrix A(n, n, 0.0);
    for (idx i = 0; i < n; ++i) {
        A(i, i) = static_cast<real>(n);
        if (i > 0)     A(i, i - 1) = 1.0;
        if (i < n - 1) A(i, i + 1) = 1.0;
    }
    Vector b(n, 1.0), x(n, 0.0);
    for (auto _ : state) {
        state.PauseTiming();
        for (idx i = 0; i < n; ++i) x[i] = 0.0;
        state.ResumeTiming();
        auto r = cg(A, b, x);
        benchmark::DoNotOptimize(x.data());
        state.counters["iters"] = static_cast<double>(r.iterations);
    }
    state.SetComplexityN(static_cast<int64_t>(n));
}
BENCHMARK(BM_CG)->RangeMultiplier(2)->Range(32, 256)->Complexity();

#ifdef NUMERICS_HAS_CUDA
static void BM_CG_GPU(benchmark::State& state) {
    idx n = static_cast<idx>(state.range(0));
    Matrix A(n, n, 0.0);
    for (idx i = 0; i < n; ++i) {
        A(i, i) = static_cast<real>(n);
        if (i > 0)     A(i, i - 1) = 1.0;
        if (i < n - 1) A(i, i + 1) = 1.0;
    }
    Vector b(n, 1.0), x(n, 0.0);
    for (auto _ : state) {
        state.PauseTiming();
        for (idx i = 0; i < n; ++i) x[i] = 0.0;
        state.ResumeTiming();
        auto r = cg(A, b, x, 1e-10, 1000, gpu);
        cudaDeviceSynchronize();
        benchmark::DoNotOptimize(x.data());
        state.counters["iters"] = static_cast<double>(r.iterations);
    }
    state.SetComplexityN(static_cast<int64_t>(n));
}
BENCHMARK(BM_CG_GPU)->RangeMultiplier(2)->Range(32, 256)->Complexity();
#endif

// Thomas algorithm (tridiagonal solver)  -- no parallel equivalent

static void BM_Thomas(benchmark::State& state) {
    idx n = static_cast<idx>(state.range(0));
    Vector a(n - 1, -1.0), b(n, 2.0), c(n - 1, -1.0), d(n, 1.0), x(n);
    for (auto _ : state) {
        thomas(a, b, c, d, x);
        benchmark::DoNotOptimize(x.data());
    }
    state.SetBytesProcessed(
        state.iterations() * static_cast<int64_t>(5 * n - 2) * sizeof(real));
    state.SetComplexityN(static_cast<int64_t>(n));
}
BENCHMARK(BM_Thomas)->RangeMultiplier(4)->Range(64, 1 << 16)->Complexity();
