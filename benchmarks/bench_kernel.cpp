/// @file bench_kernel.cpp
/// @brief Benchmarks for the kernel module (Tier 1, 2, 3).
///
/// Three groups:
///   Tier 2 array/reduce  -- kernel::array::axpby and kernel::reduce::l1_norm
///                           (seq_t, par_t) vs equivalent manual scalar loops.
///   Tier 3 subspace      -- mgs_orthogonalize with a vector-basis (GMRES path)
///                           and a matrix-basis (Lanczos path), each compared
///                           against the old manual nested loops they replaced.
///   End-to-end           -- Full arnoldi_step (the GMRES inner loop) vs the
///                           inline code that krylov.cpp used before the refactor.
///
/// Run just these benchmarks:
///   ./numerics_bench --benchmark_filter=BM_Kernel

#include <benchmark/benchmark.h>
#include "numerics.hpp"
#include "kernel/kernel.hpp"
#include <cmath>
#include <vector>

using namespace num;
namespace ks = num::kernel::subspace;
namespace ka = num::kernel::array;
namespace kr = num::kernel::reduce;

// ============================================================================
// Tier-2: array::axpby  (y = a*x + b*y)
// ============================================================================

static void BM_Kernel_Axpby_Manual(benchmark::State& state) {
    idx n = state.range(0);
    Vector x(n, 1.0), y(n, 2.0);
    const real a = 0.5, b = 1.5;
    for (auto _ : state) {
        const real* xp = x.data();
        real*       yp = y.data();
        for (idx i = 0; i < n; ++i) { yp[i] = (a * xp[i]) + (b * yp[i]); }
        benchmark::DoNotOptimize(yp);
    }
    state.SetBytesProcessed(state.iterations() * 3 * n * sizeof(real));
}
BENCHMARK(BM_Kernel_Axpby_Manual)->RangeMultiplier(4)->Range(1<<12, 1<<20);

static void BM_Kernel_Axpby_Seq(benchmark::State& state) {
    idx n = state.range(0);
    Vector x(n, 1.0), y(n, 2.0);
    for (auto _ : state) {
        ka::axpby(real(0.5), x, real(1.5), y, kernel::kseq);
        benchmark::DoNotOptimize(y.data());
    }
    state.SetBytesProcessed(state.iterations() * 3 * n * sizeof(real));
}
BENCHMARK(BM_Kernel_Axpby_Seq)->RangeMultiplier(4)->Range(1<<12, 1<<20);

static void BM_Kernel_Axpby_Par(benchmark::State& state) {
    idx n = state.range(0);
    Vector x(n, 1.0), y(n, 2.0);
    for (auto _ : state) {
        ka::axpby(real(0.5), x, real(1.5), y, kernel::kpar);
        benchmark::DoNotOptimize(y.data());
    }
    state.SetBytesProcessed(state.iterations() * 3 * n * sizeof(real));
}
BENCHMARK(BM_Kernel_Axpby_Par)->RangeMultiplier(4)->Range(1<<12, 1<<20);

// ============================================================================
// Tier-2: reduce::l1_norm
// ============================================================================

static void BM_Kernel_L1Norm_Manual(benchmark::State& state) {
    idx n = state.range(0);
    Vector x(n, 1.0);
    for (auto _ : state) {
        real s = 0.0;
        const real* xp = x.data();
        for (idx i = 0; i < n; ++i) { s += std::abs(xp[i]); }
        benchmark::DoNotOptimize(s);
    }
    state.SetBytesProcessed(state.iterations() * n * sizeof(real));
}
BENCHMARK(BM_Kernel_L1Norm_Manual)->RangeMultiplier(4)->Range(1<<12, 1<<20);

static void BM_Kernel_L1Norm_Seq(benchmark::State& state) {
    idx n = state.range(0);
    Vector x(n, 1.0);
    for (auto _ : state) {
        real s = kr::l1_norm(x, kernel::kseq);
        benchmark::DoNotOptimize(s);
    }
    state.SetBytesProcessed(state.iterations() * n * sizeof(real));
}
BENCHMARK(BM_Kernel_L1Norm_Seq)->RangeMultiplier(4)->Range(1<<12, 1<<20);

static void BM_Kernel_L1Norm_Par(benchmark::State& state) {
    idx n = state.range(0);
    Vector x(n, 1.0);
    for (auto _ : state) {
        real s = kr::l1_norm(x, kernel::kpar);
        benchmark::DoNotOptimize(s);
    }
    state.SetBytesProcessed(state.iterations() * n * sizeof(real));
}
BENCHMARK(BM_Kernel_L1Norm_Par)->RangeMultiplier(4)->Range(1<<12, 1<<20);

// ============================================================================
// Tier-3 subspace: mgs_orthogonalize  (vector-basis, the GMRES inner path)
// k vectors of length n, orthogonalize a new vector against all of them.
// ============================================================================

static void BM_Kernel_MgsVec_Manual(benchmark::State& state) {
    idx n = state.range(0);
    idx k = 30; // typical GMRES restart size
    std::vector<Vector> basis;
    basis.reserve(k);
    for (idx i = 0; i < k; ++i) {
        basis.emplace_back(n, 0.0);
        basis.back()[i % n] = 1.0;
    }
    Vector v(n, 1.0);
    std::vector<real> h(k + 1, 0.0);

    for (auto _ : state) {
        // old inline code: k passes of dot + axpy
        for (idx i = 0; i < k; ++i) {
            real hi = 0.0;
            for (idx j = 0; j < n; ++j) { hi += v[j] * basis[i][j]; }
            h[i] = hi;
            for (idx j = 0; j < n; ++j) { v[j] -= hi * basis[i][j]; }
        }
        benchmark::DoNotOptimize(v.data());
        // restore v so the next iteration has the same input
        for (idx j = 0; j < n; ++j) { v[j] = 1.0; }
    }
}
BENCHMARK(BM_Kernel_MgsVec_Manual)->RangeMultiplier(4)->Range(1<<10, 1<<16);

static void BM_Kernel_MgsVec_Kernel(benchmark::State& state) {
    idx n = state.range(0);
    idx k = 30;
    std::vector<Vector> basis;
    basis.reserve(k);
    for (idx i = 0; i < k; ++i) {
        basis.emplace_back(n, 0.0);
        basis.back()[i % n] = 1.0;
    }
    Vector v(n, 1.0);
    std::vector<real> h(k + 1, 0.0);

    for (auto _ : state) {
        real beta = ks::mgs_orthogonalize(basis, v, h, k);
        benchmark::DoNotOptimize(beta);
        benchmark::DoNotOptimize(v.data());
        for (idx j = 0; j < n; ++j) { v[j] = 1.0; }
    }
}
BENCHMARK(BM_Kernel_MgsVec_Kernel)->RangeMultiplier(4)->Range(1<<10, 1<<16);

// ============================================================================
// Tier-3 subspace: mgs_orthogonalize  (matrix-basis, the Lanczos inner path)
// Basis stored as columns of a row-major Matrix.
// ============================================================================

static void BM_Kernel_MgsMat_Manual(benchmark::State& state) {
    idx n = state.range(0);
    idx k = 30;
    Matrix basis(n, k, 0.0);
    for (idx l = 0; l < k; ++l) { basis(l % n, l) = 1.0; }
    Vector v(n, 1.0);

    for (auto _ : state) {
        for (idx l = 0; l < k; ++l) {
            real proj = 0.0;
            for (idx i = 0; i < n; ++i) { proj += basis(i, l) * v[i]; }
            for (idx i = 0; i < n; ++i) { v[i] -= proj * basis(i, l); }
        }
        benchmark::DoNotOptimize(v.data());
        for (idx j = 0; j < n; ++j) { v[j] = 1.0; }
    }
}
BENCHMARK(BM_Kernel_MgsMat_Manual)->RangeMultiplier(4)->Range(1<<10, 1<<16);

static void BM_Kernel_MgsMat_Kernel(benchmark::State& state) {
    idx n = state.range(0);
    idx k = 30;
    Matrix basis(n, k, 0.0);
    for (idx l = 0; l < k; ++l) { basis(l % n, l) = 1.0; }
    Vector v(n, 1.0);

    for (auto _ : state) {
        real beta = ks::mgs_orthogonalize(basis, k, v);
        benchmark::DoNotOptimize(beta);
        benchmark::DoNotOptimize(v.data());
        for (idx j = 0; j < n; ++j) { v[j] = 1.0; }
    }
}
BENCHMARK(BM_Kernel_MgsMat_Kernel)->RangeMultiplier(4)->Range(1<<10, 1<<16);

// ============================================================================
// End-to-end: full Arnoldi step  (inline vs kernel::subspace::arnoldi_step)
// Simulates one restart cycle of GMRES(30) on an n-dimensional problem.
// Inline version mirrors the exact code that krylov.cpp had before the refactor.
// ============================================================================

// Operator: y[i] = (i+1) * x[i]  (diagonal with distinct eigenvalues).
// This generates a full Krylov basis of dimension >= restart, so no
// lucky breakdown, and every emplace_back inside arnoldi_step fires.

// Operator: y[i] = (i+1) * x[i]  (distinct diagonal eigenvalues).
// Start vector: uniform 1/sqrt(n) — not an eigenvector, so the Krylov subspace
// has dimension min(restart, n) and no lucky breakdown for n >= restart.

static void BM_Kernel_Arnoldi_Inline(benchmark::State& state) {
    idx n       = state.range(0);
    idx restart = 30;
    const real v0 = real(1) / std::sqrt(static_cast<real>(n));

    for (auto _ : state) {
        std::vector<Vector> V;
        V.reserve(restart + 1);
        V.emplace_back(n, v0);   // uniform start vector, already normalised
        std::vector<real> h(restart + 1, 0.0);

        for (idx j = 0; j < restart; ++j) {
            Vector w(n);
            for (idx i = 0; i < n; ++i) {
                w[i] = static_cast<real>(i + 1) * V[j][i];
            }
            for (idx i = 0; i <= j; ++i) {
                real hi = 0.0;
                for (idx k = 0; k < n; ++k) { hi += w[k] * V[i][k]; }
                h[i] = hi;
                for (idx k = 0; k < n; ++k) { w[k] -= hi * V[i][k]; }
            }
            real h_next = 0.0;
            for (idx k = 0; k < n; ++k) { h_next += w[k] * w[k]; }
            h_next = std::sqrt(h_next);
            h[j + 1] = h_next;

            if (h_next > real(1e-15)) {
                V.emplace_back(n);
                for (idx k = 0; k < n; ++k) { V[j + 1][k] = w[k] / h_next; }
            } else {
                break;
            }
        }
        benchmark::DoNotOptimize(V.back().data());
    }
}
BENCHMARK(BM_Kernel_Arnoldi_Inline)->RangeMultiplier(4)->Range(1<<8, 1<<14);

static void BM_Kernel_Arnoldi_Kernel(benchmark::State& state) {
    idx n       = state.range(0);
    idx restart = 30;
    const real v0 = real(1) / std::sqrt(static_cast<real>(n));

    for (auto _ : state) {
        std::vector<Vector> V;
        V.reserve(restart + 1);
        V.emplace_back(n, v0);
        std::vector<real> h(restart + 1, 0.0);

        auto A_op = ks::make_op(
            [](const Vector& x, Vector& y) {
                // y pre-allocated by CallableOp::apply; no heap alloc in hot path
                for (idx i = 0; i < x.size(); ++i) {
                    y[i] = static_cast<real>(i + 1) * x[i];
                }
            }, n);

        Vector scratch(n);
        for (idx j = 0; j < restart; ++j) {
            real beta = ks::arnoldi_step(A_op, V, h, j, scratch, real(1e-15));
            benchmark::DoNotOptimize(beta);
            if (beta <= real(1e-15)) { break; }
        }
        benchmark::DoNotOptimize(V.back().data());
    }
}
BENCHMARK(BM_Kernel_Arnoldi_Kernel)->RangeMultiplier(4)->Range(1<<8, 1<<14);
