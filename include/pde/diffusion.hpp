/// @file pde/diffusion.hpp
/// @brief Diffusion operators and implicit system builders for 2D grids.
///
/// Explicit Euler stencil steps (periodic or Dirichlet BCs):
///   diffusion_step_2d            -- periodic
///   diffusion_step_2d_dirichlet  -- Dirichlet, 2nd-order
///   diffusion_step_2d_4th_dirichlet -- Dirichlet, 4th-order
///
/// Implicit system builders (backward Euler):
///   laplacian_sparse_2d          -- bare N^2 x N^2 Laplacian matrix
///   backward_euler_matrix        -- A = I - coeff*L  (sparse, SPD)
///
/// Solver factory:
///   make_cg_solver(A)            -- returns a LinearSolver using matrix-free CG
///
/// Time integration lives in ode/implicit.hpp: num::ode::advance(u, solver, p).
#pragma once

#include "pde/stencil.hpp"
#include "core/vector.hpp"
#include "core/policy.hpp"
#include "fields/grid2d.hpp"
#include "linalg/sparse/sparse.hpp"
#include "linalg/solvers/cg.hpp"
#include "linalg/solvers/linear_solver.hpp"

namespace num::pde {

/// Explicit Euler diffusion step on a periodic 2D grid:
///   u += coeff * Lap_periodic(u)
inline void diffusion_step_2d(Vector& u, int N, double coeff,
                               Backend b = best_backend) {
    Vector lap(u.size());
    laplacian_stencil_2d_periodic(u, lap, N);
    axpy(coeff, lap, u, b);
}

/// Explicit Euler diffusion step with Dirichlet (zero) BCs:
///   u += coeff * Lap_dirichlet(u)
inline void diffusion_step_2d_dirichlet(Vector& u, int N, double coeff,
                                         Backend b = best_backend) {
    Vector lap(u.size());
    laplacian_stencil_2d(u, lap, N);
    axpy(coeff, lap, u, b);
}

/// Explicit Euler diffusion step with Dirichlet BCs and 4th-order Laplacian.
/// Stability requires coeff <= ~0.08 (vs 0.25 for 2nd order).
inline void diffusion_step_2d_4th_dirichlet(Vector& u, int N, double coeff,
                                              Backend b = best_backend) {
    Vector lap(u.size());
    laplacian_stencil_2d_4th(u, lap, N);
    axpy(coeff, lap, u, b);
}

// ScalarField2D overloads

inline void diffusion_step_2d_dirichlet(ScalarField2D& g, double coeff,
                                         Backend b = best_backend) {
    diffusion_step_2d_dirichlet(g.vec(), g.N(), coeff, b);
}

inline void diffusion_step_2d_4th_dirichlet(ScalarField2D& g, double coeff,
                                              Backend b = best_backend) {
    diffusion_step_2d_4th_dirichlet(g.vec(), g.N(), coeff, b);
}

/// Build the N^2 x N^2 sparse 5-point Laplacian matrix (Dirichlet BCs).
/// Entry (k,k) = -4, off-diagonals = +1 for each interior neighbor.
inline SparseMatrix laplacian_sparse_2d(int N) {
    const int n = N * N;
    std::vector<idx>  rows, cols;
    std::vector<real> vals;
    rows.reserve(5 * n); cols.reserve(5 * n); vals.reserve(5 * n);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int k = i * N + j;
            rows.push_back(k); cols.push_back(k); vals.push_back(-4.0);
            if (i > 0)   { rows.push_back(k); cols.push_back((i-1)*N+j); vals.push_back(1.0); }
            if (i < N-1) { rows.push_back(k); cols.push_back((i+1)*N+j); vals.push_back(1.0); }
            if (j > 0)   { rows.push_back(k); cols.push_back(i*N+(j-1)); vals.push_back(1.0); }
            if (j < N-1) { rows.push_back(k); cols.push_back(i*N+(j+1)); vals.push_back(1.0); }
        }
    }
    return SparseMatrix::from_triplets(n, n, rows, cols, vals);
}

/// Build the backward Euler system matrix A = I - coeff*L.
/// Solve A * u^{n+1} = u^n at each time step.
inline SparseMatrix backward_euler_matrix(int N, double coeff) {
    const int n = N * N;
    std::vector<idx>  rows, cols;
    std::vector<real> vals;
    rows.reserve(5 * n); cols.reserve(5 * n); vals.reserve(5 * n);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int k = i * N + j;
            rows.push_back(k); cols.push_back(k); vals.push_back(1.0 + 4.0 * coeff);
            if (i > 0)   { rows.push_back(k); cols.push_back((i-1)*N+j); vals.push_back(-coeff); }
            if (i < N-1) { rows.push_back(k); cols.push_back((i+1)*N+j); vals.push_back(-coeff); }
            if (j > 0)   { rows.push_back(k); cols.push_back(i*N+(j-1)); vals.push_back(-coeff); }
            if (j < N-1) { rows.push_back(k); cols.push_back(i*N+(j+1)); vals.push_back(-coeff); }
        }
    }
    return SparseMatrix::from_triplets(n, n, rows, cols, vals);
}

/// Grid2D overload -- N is read from the grid.
inline SparseMatrix backward_euler_matrix(const Grid2D& grid, double coeff) {
    return backward_euler_matrix(grid.N, coeff);
}

/// Returns a LinearSolver that solves A*x = rhs using matrix-free CG.
/// Captures A by reference -- A must outlive the returned callable.
inline LinearSolver make_cg_solver(const SparseMatrix& A, real tol = 1e-6) {
    return [&A, tol](const Vector& rhs, Vector& x) {
        auto matvec = [&A](const Vector& v, Vector& w) { sparse_matvec(A, v, w); };
        return cg_matfree(matvec, rhs, x, tol);
    };
}

} // namespace num::pde

// Lift make_cg_solver into num:: so callers can write num::make_cg_solver(A).
namespace num {
    using pde::make_cg_solver;
}
