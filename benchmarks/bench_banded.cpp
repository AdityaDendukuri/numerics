/// @file bench_banded.cpp
/// @brief Benchmarks for banded matrix solver
///
/// Measures performance characteristics for HPC applications:
///   - Throughput (FLOPS) for different matrix sizes
///   - Scaling behavior with bandwidth
///   - Memory bandwidth utilization
///   - Comparison with tridiagonal solver

#include <benchmark/benchmark.h>
#include "linalg/banded/banded.hpp"
#include "linalg/solvers/solvers.hpp"
#include "linalg/factorization/thomas.hpp"
#include <cmath>
#include <cstring>
#include <random>

using namespace num;

// Helper Functions

static void setup_tridiagonal(BandedMatrix& A, idx n) {
    // 1D Laplacian pattern: -1, 2, -1
    for (idx i = 0; i < n; ++i) {
        A(i, i) = 2.0;
        if (i > 0) A(i, i - 1) = -1.0;
        if (i < n - 1) A(i, i + 1) = -1.0;
    }
}

static void setup_pentadiagonal(BandedMatrix& A, idx n) {
    // Biharmonic-like pattern with diagonal dominance
    for (idx i = 0; i < n; ++i) {
        A(i, i) = 10.0;
        if (i > 0) A(i, i - 1) = -2.0;
        if (i > 1) A(i, i - 2) = -0.5;
        if (i < n - 1) A(i, i + 1) = -2.0;
        if (i < n - 2) A(i, i + 2) = -0.5;
    }
}

static void setup_general_banded(BandedMatrix& A, idx n, idx kl, idx ku) {
    // Diagonally dominant general banded matrix
    for (idx j = 0; j < n; ++j) {
        real off_diag_sum = 0.0;
        for (idx i = (j > ku ? j - ku : 0); i < j; ++i) {
            A(i, j) = -0.1;
            off_diag_sum += 0.1;
        }
        for (idx i = j + 1; i <= std::min(j + kl, n - 1); ++i) {
            A(i, j) = -0.1;
            off_diag_sum += 0.1;
        }
        A(j, j) = off_diag_sum + 2.0;
    }
}

// Tridiagonal Solver Benchmarks

static void BM_BandedSolve_Tridiagonal(benchmark::State& state) {
    idx n = state.range(0);

    BandedMatrix A(n, 1, 1, 0.0);
    setup_tridiagonal(A, n);

    Vector b(n, 1.0);
    Vector x(n, 0.0);

    for (auto _ : state) {
        state.PauseTiming();
        for (idx i = 0; i < n; ++i) x[i] = 0.0;
        state.ResumeTiming();

        banded_solve(A, b, x);
        benchmark::DoNotOptimize(x.data());
    }

    // Report metrics
    // Tridiagonal LU: ~5n ops for factorization, ~3n for solve
    idx flops = 8 * n;
    state.SetItemsProcessed(state.iterations() * n);
    state.counters["FLOPS"] = benchmark::Counter(
        state.iterations() * flops,
        benchmark::Counter::kIsRate);
}
BENCHMARK(BM_BandedSolve_Tridiagonal)
    ->RangeMultiplier(4)
    ->Range(64, 1 << 18)
    ->Unit(benchmark::kMicrosecond);

// Compare with Thomas algorithm
static void BM_Thomas_Baseline(benchmark::State& state) {
    idx n = state.range(0);

    Vector a(n - 1, -1.0);  // Lower diagonal
    Vector b(n, 2.0);       // Main diagonal
    Vector c(n - 1, -1.0);  // Upper diagonal
    Vector d(n, 1.0);       // RHS
    Vector x(n);

    for (auto _ : state) {
        thomas(a, b, c, d, x);
        benchmark::DoNotOptimize(x.data());
    }

    idx flops = 8 * n;
    state.SetItemsProcessed(state.iterations() * n);
    state.counters["FLOPS"] = benchmark::Counter(
        state.iterations() * flops,
        benchmark::Counter::kIsRate);
}
BENCHMARK(BM_Thomas_Baseline)
    ->RangeMultiplier(4)
    ->Range(64, 1 << 18)
    ->Unit(benchmark::kMicrosecond);

// Pentadiagonal Solver Benchmarks

static void BM_BandedSolve_Pentadiagonal(benchmark::State& state) {
    idx n = state.range(0);

    BandedMatrix A(n, 2, 2, 0.0);
    setup_pentadiagonal(A, n);

    Vector b(n, 1.0);
    Vector x(n, 0.0);

    for (auto _ : state) {
        state.PauseTiming();
        for (idx i = 0; i < n; ++i) x[i] = 0.0;
        state.ResumeTiming();

        banded_solve(A, b, x);
        benchmark::DoNotOptimize(x.data());
    }

    // Pentadiagonal: ~25n ops
    idx flops = 25 * n;
    state.SetItemsProcessed(state.iterations() * n);
    state.counters["FLOPS"] = benchmark::Counter(
        state.iterations() * flops,
        benchmark::Counter::kIsRate);
}
BENCHMARK(BM_BandedSolve_Pentadiagonal)
    ->RangeMultiplier(4)
    ->Range(64, 1 << 16)
    ->Unit(benchmark::kMicrosecond);

// General Banded Solver Benchmarks

static void BM_BandedSolve_General_KL2_KU4(benchmark::State& state) {
    idx n = state.range(0);
    idx kl = 2, ku = 4;

    BandedMatrix A(n, kl, ku, 0.0);
    setup_general_banded(A, n, kl, ku);

    Vector b(n, 1.0);
    Vector x(n, 0.0);

    for (auto _ : state) {
        state.PauseTiming();
        for (idx i = 0; i < n; ++i) x[i] = 0.0;
        state.ResumeTiming();

        banded_solve(A, b, x);
        benchmark::DoNotOptimize(x.data());
    }

    // General banded: O(n * kl * (kl + ku))
    idx flops = n * kl * (kl + ku) * 2;
    state.SetItemsProcessed(state.iterations() * n);
    state.counters["FLOPS"] = benchmark::Counter(
        state.iterations() * flops,
        benchmark::Counter::kIsRate);
}
BENCHMARK(BM_BandedSolve_General_KL2_KU4)
    ->RangeMultiplier(4)
    ->Range(64, 1 << 16)
    ->Unit(benchmark::kMicrosecond);

static void BM_BandedSolve_General_KL5_KU5(benchmark::State& state) {
    idx n = state.range(0);
    idx kl = 5, ku = 5;

    BandedMatrix A(n, kl, ku, 0.0);
    setup_general_banded(A, n, kl, ku);

    Vector b(n, 1.0);
    Vector x(n, 0.0);

    for (auto _ : state) {
        state.PauseTiming();
        for (idx i = 0; i < n; ++i) x[i] = 0.0;
        state.ResumeTiming();

        banded_solve(A, b, x);
        benchmark::DoNotOptimize(x.data());
    }

    idx flops = n * kl * (kl + ku) * 2;
    state.SetItemsProcessed(state.iterations() * n);
    state.counters["FLOPS"] = benchmark::Counter(
        state.iterations() * flops,
        benchmark::Counter::kIsRate);
}
BENCHMARK(BM_BandedSolve_General_KL5_KU5)
    ->RangeMultiplier(4)
    ->Range(64, 1 << 14)
    ->Unit(benchmark::kMicrosecond);

// LU Factorization vs Solve Benchmarks

static void BM_BandedLU_Factorization(benchmark::State& state) {
    idx n = state.range(0);

    BandedMatrix A_template(n, 2, 2, 0.0);
    setup_pentadiagonal(A_template, n);

    std::unique_ptr<idx[]> ipiv = std::make_unique<idx[]>(n);

    for (auto _ : state) {
        state.PauseTiming();
        BandedMatrix A = A_template;  // Fresh copy each iteration
        state.ResumeTiming();

        banded_lu(A, ipiv.get());
        benchmark::DoNotOptimize(A.data());
    }

    state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(BM_BandedLU_Factorization)
    ->RangeMultiplier(4)
    ->Range(64, 1 << 16)
    ->Unit(benchmark::kMicrosecond);

static void BM_BandedLU_Solve(benchmark::State& state) {
    idx n = state.range(0);

    BandedMatrix A(n, 2, 2, 0.0);
    setup_pentadiagonal(A, n);

    std::unique_ptr<idx[]> ipiv = std::make_unique<idx[]>(n);
    banded_lu(A, ipiv.get());  // Factor once

    Vector b(n, 1.0);

    for (auto _ : state) {
        state.PauseTiming();
        Vector x = b;  // Copy RHS
        state.ResumeTiming();

        banded_lu_solve(A, ipiv.get(), x);
        benchmark::DoNotOptimize(x.data());
    }

    state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(BM_BandedLU_Solve)
    ->RangeMultiplier(4)
    ->Range(64, 1 << 16)
    ->Unit(benchmark::kMicrosecond);

// Multiple RHS Benchmarks

static void BM_BandedSolve_MultiRHS(benchmark::State& state) {
    idx n = state.range(0);
    idx nrhs = 16;  // Common for radiative transfer (spectral bands)

    BandedMatrix A(n, 2, 2, 0.0);
    setup_pentadiagonal(A, n);

    std::unique_ptr<idx[]> ipiv = std::make_unique<idx[]>(n);
    banded_lu(A, ipiv.get());

    std::unique_ptr<real[]> B = std::make_unique<real[]>(n * nrhs);
    for (idx i = 0; i < n * nrhs; ++i) B[i] = 1.0;

    std::unique_ptr<real[]> B_work = std::make_unique<real[]>(n * nrhs);

    for (auto _ : state) {
        state.PauseTiming();
        std::memcpy(B_work.get(), B.get(), n * nrhs * sizeof(real));
        state.ResumeTiming();

        banded_lu_solve_multi(A, ipiv.get(), B_work.get(), nrhs);
        benchmark::DoNotOptimize(B_work.get());
    }

    state.SetItemsProcessed(state.iterations() * n * nrhs);
    state.counters["RHS"] = nrhs;
}
BENCHMARK(BM_BandedSolve_MultiRHS)
    ->RangeMultiplier(4)
    ->Range(64, 1 << 14)
    ->Unit(benchmark::kMicrosecond);

// Matrix-Vector Product Benchmarks

static void BM_BandedMatvec_Tridiagonal(benchmark::State& state) {
    idx n = state.range(0);

    BandedMatrix A(n, 1, 1, 0.0);
    setup_tridiagonal(A, n);

    Vector x(n, 1.0);
    Vector y(n);

    for (auto _ : state) {
        banded_matvec(A, x, y);
        benchmark::DoNotOptimize(y.data());
    }

    // 3 multiplies + 2 adds per row for tridiagonal
    idx flops = 5 * n;
    state.SetItemsProcessed(state.iterations() * n);
    state.counters["FLOPS"] = benchmark::Counter(
        state.iterations() * flops,
        benchmark::Counter::kIsRate);
    state.SetBytesProcessed(state.iterations() * (A.ldab() * n + 2 * n) * sizeof(real));
}
BENCHMARK(BM_BandedMatvec_Tridiagonal)
    ->RangeMultiplier(4)
    ->Range(64, 1 << 18)
    ->Unit(benchmark::kMicrosecond);

static void BM_BandedMatvec_Pentadiagonal(benchmark::State& state) {
    idx n = state.range(0);

    BandedMatrix A(n, 2, 2, 0.0);
    setup_pentadiagonal(A, n);

    Vector x(n, 1.0);
    Vector y(n);

    for (auto _ : state) {
        banded_matvec(A, x, y);
        benchmark::DoNotOptimize(y.data());
    }

    // 5 multiplies + 4 adds per row for pentadiagonal
    idx flops = 9 * n;
    state.SetItemsProcessed(state.iterations() * n);
    state.counters["FLOPS"] = benchmark::Counter(
        state.iterations() * flops,
        benchmark::Counter::kIsRate);
    state.SetBytesProcessed(state.iterations() * (A.ldab() * n + 2 * n) * sizeof(real));
}
BENCHMARK(BM_BandedMatvec_Pentadiagonal)
    ->RangeMultiplier(4)
    ->Range(64, 1 << 18)
    ->Unit(benchmark::kMicrosecond);

// Bandwidth Scaling Benchmark

static void BM_BandedSolve_Bandwidth_Scaling(benchmark::State& state) {
    idx n = 4096;  // Fixed size
    idx bandwidth = state.range(0);
    idx kl = bandwidth / 2;
    idx ku = bandwidth - kl;

    BandedMatrix A(n, kl, ku, 0.0);
    setup_general_banded(A, n, kl, ku);

    Vector b(n, 1.0);
    Vector x(n, 0.0);

    for (auto _ : state) {
        state.PauseTiming();
        for (idx i = 0; i < n; ++i) x[i] = 0.0;
        state.ResumeTiming();

        banded_solve(A, b, x);
        benchmark::DoNotOptimize(x.data());
    }

    state.SetItemsProcessed(state.iterations() * n);
    state.counters["bandwidth"] = bandwidth;
}
BENCHMARK(BM_BandedSolve_Bandwidth_Scaling)
    ->Arg(3)    // Tridiagonal
    ->Arg(5)    // Pentadiagonal
    ->Arg(7)
    ->Arg(11)
    ->Arg(15)
    ->Arg(21)
    ->Unit(benchmark::kMicrosecond);

// Radiative Transfer Application Benchmark

// Simulates the matrix structure used in two-stream radiative transfer
// equations (relevant for TUVX photolysis solver)
static void BM_RadiativeTransfer_TwoStream(benchmark::State& state) {
    idx n_layers = state.range(0);  // Number of atmospheric layers
    idx n = 2 * n_layers;  // 2 unknowns per layer (up/down flux)

    // Two-stream approximation gives a tridiagonal-like structure
    // but with 2x2 block structure
    BandedMatrix A(n, 2, 2, 0.0);

    // Setup typical radiative transfer matrix structure
    for (idx layer = 0; layer < n_layers; ++layer) {
        idx i = 2 * layer;

        // Diagonal blocks (transmission/reflection within layer)
        A(i, i) = 1.0 + 0.1;  // 1 + optical depth term
        A(i + 1, i + 1) = 1.0 + 0.1;
        A(i, i + 1) = -0.05;  // Coupling between up/down
        A(i + 1, i) = -0.05;

        // Off-diagonal blocks (coupling to adjacent layers)
        if (layer > 0) {
            A(i, i - 2) = -0.08;
            A(i + 1, i - 1) = -0.08;
        }
        if (layer < n_layers - 1) {
            A(i, i + 2) = -0.08;
            A(i + 1, i + 3) = -0.08;
        }
    }

    Vector b(n, 1.0);  // Solar source term
    Vector x(n, 0.0);

    for (auto _ : state) {
        state.PauseTiming();
        for (idx i = 0; i < n; ++i) x[i] = 0.0;
        state.ResumeTiming();

        banded_solve(A, b, x);
        benchmark::DoNotOptimize(x.data());
    }

    state.SetItemsProcessed(state.iterations() * n);
    state.counters["layers"] = n_layers;
}
BENCHMARK(BM_RadiativeTransfer_TwoStream)
    ->Arg(50)    // Typical vertical resolution
    ->Arg(100)
    ->Arg(200)
    ->Arg(500)
    ->Arg(1000)  // High resolution
    ->Unit(benchmark::kMicrosecond);

// Note: BENCHMARK_MAIN() is defined in bench_linalg.cpp
