/// @file bench_eigen.cpp
/// @brief 3-way benchmark: Jacobi eig vs Lanczos vs LAPACK dsyevd.
///
/// Variants:
///   backend::seq    -- our cyclic Jacobi (full spectrum, O(n^3))
///   backend::omp    -- our Jacobi with OpenMP rotation loops
///   backend::lapack -- LAPACKE_dsyevd divide-and-conquer (full spectrum)
///
/// Lanczos uses the operator protocol and targets only a few eigenvalues
/// (k=10), so it is on its own benchmark rather than the 3-way template.
///
/// Run with:
///   ./numerics_bench --benchmark_filter=BM_Eig

#include "linear/eigen/eigen.hpp"
#include "container/matrix_ops.hpp"
#include "numerics.hpp"
#include <benchmark/benchmark.h>

using namespace num;

static Matrix make_sym(idx n) {
    Matrix A(n, n, 0.0);
    for (idx i = 0; i < n; ++i)
        for (idx j = i; j < n; ++j) {
            real v = static_cast<real>(1) / static_cast<real>(1 + i + j);
            A(i, j) = A(j, i) = v;
        }
    // ensure positive eigenvalues
    for (idx i = 0; i < n; ++i)
        A(i, i) += static_cast<real>(n);
    return A;
}

// ── Full symmetric eigendecomposition ────────────────────────────────────────

template <Backend B>
static void BM_EigSym(benchmark::State &state) {
    idx n = static_cast<idx>(state.range(0));
    Matrix A = make_sym(n);
    for (auto _ : state) {
        auto r = eig_sym(assume_symmetric(A), 1e-12, 100, B);
        benchmark::DoNotOptimize(r.values.data());
    }
    state.counters["GFLOP/s"] = benchmark::Counter(
        4.0 / 3.0 * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n),
        benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.SetComplexityN(static_cast<int64_t>(n));
}
BENCHMARK_TEMPLATE(BM_EigSym, backend::seq)->RangeMultiplier(2)->Range(32, 512)->Complexity();
// NOTE: backend::omp Jacobi is intentionally excluded from the benchmark.
// Cyclic Jacobi applies one OpenMP parallel region per rotation, meaning
// O(n^2) team launches per sweep × O(n) sweeps = O(n^3) launches total.
// At n=128 this is ~800K launches; thread overhead (~10µs each) dwarfs the
// actual work (~0.5µs per inner loop), making OMP Jacobi slower than seq for
// all practical n.  Correct OMP Jacobi requires tournament-ordered batches
// (Brent-Luk-Van Loan) to amortize the team launch across O(n) independent
// rotations at once — a future improvement.  The OMP backend remains correct
// and available; it just should not be benchmarked against seq this way.
#if defined(NUMERICS_HAS_LAPACK)
BENCHMARK_TEMPLATE(BM_EigSym, backend::lapack)->RangeMultiplier(2)->Range(32, 512)->Complexity();
#endif

static void BM_Lanczos(benchmark::State &state) {
    idx n = static_cast<idx>(state.range(0));
    constexpr idx k = 10;
    Matrix A = make_sym(n);
    auto op =
        operators::make_op([&](const Vector &v, Vector &w) { matvec(A, v, w, backend::dflt); }, n);
    auto Aop = operators::assume_symmetric(op);
    for (auto _ : state) {
        auto r = lanczos(Aop, k, 1e-10, 0, backend::seq);
        benchmark::DoNotOptimize(r.ritz_values.data());
    }
    state.SetComplexityN(static_cast<int64_t>(n));
}
BENCHMARK(BM_Lanczos)->RangeMultiplier(2)->Range(64, 2048)->Complexity();
