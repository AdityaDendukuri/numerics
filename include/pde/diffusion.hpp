/// @file pde/diffusion.hpp
/// @brief Explicit Euler diffusion steps for 2D uniform grids.
///
/// diffusion_step_2d           -- u += coeff * Lap_periodic(u),  periodic BC
/// diffusion_step_2d_dirichlet -- u += coeff * Lap_dirichlet(u), Dirichlet BC
///
/// coeff = alpha*dt/h^2 where alpha is the diffusion coefficient.
///
/// Typical usage (viscosity in Navier-Stokes):
/// @code
///   const double coeff = dt * nu / (h * h);
///   num::pde::diffusion_step_2d(u_star, N, coeff, num::best_backend);
///   num::pde::diffusion_step_2d(v_star, N, coeff, num::best_backend);
/// @endcode
#pragma once

#include "pde/stencil.hpp"
#include "core/vector.hpp"
#include "core/policy.hpp"
#include "linalg/sparse/sparse.hpp"
#include "linalg/solvers/cg.hpp"
#include "linalg/solvers/solver_result.hpp"
#include <functional>
#include <vector>

namespace num::pde {

/// Explicit Euler diffusion step on a periodic 2D grid:
///   u += coeff * Lap_periodic(u)
///
/// @param u      NxN field vector (row-major)
/// @param N      Grid side length
/// @param coeff  Diffusion coefficient * dt / h^2 (e.g. nu*dt/h^2 for
/// viscosity)
/// @param b      Backend for the axpy accumulation
inline void diffusion_step_2d(Vector& u,
                              int     N,
                              double  coeff,
                              Backend b = best_backend) {
    Vector lap(u.size());
    laplacian_stencil_2d_periodic(u, lap, N);
    axpy(coeff, lap, u, b);
}

/// Explicit Euler diffusion step with Dirichlet (zero) BCs:
///   u += coeff * Lap_dirichlet(u)
inline void diffusion_step_2d_dirichlet(Vector& u,
                                        int     N,
                                        double  coeff,
                                        Backend b = best_backend) {
    Vector lap(u.size());
    laplacian_stencil_2d(u, lap, N);
    axpy(coeff, lap, u, b);
}

/// Explicit Euler diffusion step with Dirichlet BCs and 4th-order Laplacian:
///   u += coeff * Lap4_dirichlet(u)
///
/// coeff = alpha*dt/h^2. Stability requires coeff <= ~0.08 (vs 0.25 for 2nd order)
/// due to the larger spectral radius of the 13-point stencil.
inline void diffusion_step_2d_4th_dirichlet(Vector& u,
                                            int     N,
                                            double  coeff,
                                            Backend b = best_backend) {
    Vector lap(u.size());
    laplacian_stencil_2d_4th(u, lap, N);
    axpy(coeff, lap, u, b);
}

// ScalarField2D overloads

inline void diffusion_step_2d_dirichlet(ScalarField2D& g, double coeff, Backend b = best_backend) {
    diffusion_step_2d_dirichlet(g.vec(), g.N(), coeff, b);
}

inline void diffusion_step_2d_4th_dirichlet(ScalarField2D& g, double coeff, Backend b = best_backend) {
    diffusion_step_2d_4th_dirichlet(g.vec(), g.N(), coeff, b);
}

/// Build the N^2 x N^2 sparse 5-point Laplacian matrix for an NxN Dirichlet
/// grid. Entry (k,k) = -4, off-diagonals = +1 for each interior neighbor.
inline SparseMatrix laplacian_sparse_2d(int N) {
    const int n = N * N;
    std::vector<idx>  rows, cols;
    std::vector<real> vals;
    rows.reserve(5 * n);
    cols.reserve(5 * n);
    vals.reserve(5 * n);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int k = i * N + j;
            rows.push_back(k); cols.push_back(k); vals.push_back(-4.0);
            if (i > 0)   { rows.push_back(k); cols.push_back((i-1)*N + j); vals.push_back(1.0); }
            if (i < N-1) { rows.push_back(k); cols.push_back((i+1)*N + j); vals.push_back(1.0); }
            if (j > 0)   { rows.push_back(k); cols.push_back(i*N + (j-1)); vals.push_back(1.0); }
            if (j < N-1) { rows.push_back(k); cols.push_back(i*N + (j+1)); vals.push_back(1.0); }
        }
    }
    return SparseMatrix::from_triplets(n, n, rows, cols, vals);
}

/// Build the backward Euler system matrix  A = I - coeff*L
/// where L = laplacian_sparse_2d(N) and coeff = kappa*dt/h^2.
/// Solve  A * u^{n+1} = u^n  at each time step.
inline SparseMatrix backward_euler_matrix_2d(int N, double coeff) {
    const int n = N * N;
    std::vector<idx>  rows, cols;
    std::vector<real> vals;
    rows.reserve(5 * n);
    cols.reserve(5 * n);
    vals.reserve(5 * n);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int k = i * N + j;
            rows.push_back(k); cols.push_back(k); vals.push_back(1.0 + 4.0 * coeff);
            if (i > 0)   { rows.push_back(k); cols.push_back((i-1)*N + j); vals.push_back(-coeff); }
            if (i < N-1) { rows.push_back(k); cols.push_back((i+1)*N + j); vals.push_back(-coeff); }
            if (j > 0)   { rows.push_back(k); cols.push_back(i*N + (j-1)); vals.push_back(-coeff); }
            if (j < N-1) { rows.push_back(k); cols.push_back(i*N + (j+1)); vals.push_back(-coeff); }
        }
    }
    return SparseMatrix::from_triplets(n, n, rows, cols, vals);
}

/// ScalarField2D overload -- N is read from the grid.
inline SparseMatrix backward_euler_matrix_2d(const ScalarField2D& g, double coeff) {
    return backward_euler_matrix_2d(g.N(), coeff);
}

} // namespace num::pde

namespace num {
/// Callable type for one solve step:  f(rhs, x) solves A*x = rhs in-place.
using SolveStep = std::function<SolverResult(const Vector&, Vector&)>;
} // namespace num

namespace num::pde {

/// Returns a SolveStep that solves A*x = rhs using matrix-free CG.
/// Captures A by reference -- A must outlive the returned callable.
inline SolveStep make_cg_solver(const SparseMatrix& A, real tol = 1e-6) {
    return [&A, tol](const Vector& rhs, Vector& x) {
        auto matvec = [&A](const Vector& v, Vector& w) { sparse_matvec(A, v, w); };
        return cg_matfree(matvec, rhs, x, tol);
    };
}

// Time integration

/// Parameters for diffusion_2d.
struct DiffusionParams {
    int    nstep; ///< Number of time steps
    double dt;    ///< Step size (used to report t to the observer)
};

/// Advance u forward in time using the provided solver.
/// observer(step, t, state) is called at step 0 (initial) and after each solve.
template<typename Observer>
inline void diffusion_2d(ScalarField2D&                u,
                         const SolveStep&       solver,
                         const DiffusionParams& params,
                         Observer&&             observer) {
    observer(0, 0.0, u);
    for (int s = 0; s < params.nstep; ++s) {
        Vector rhs = u.vec();
        solver(rhs, u.vec());
        observer(s + 1, (s + 1) * params.dt, u);
    }
}

/// Overload without observer.
inline void diffusion_2d(ScalarField2D&                u,
                         const SolveStep&       solver,
                         const DiffusionParams& params) {
    for (int s = 0; s < params.nstep; ++s) {
        Vector rhs = u.vec();
        solver(rhs, u.vec());
    }
}

} // namespace num::pde
