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

} // namespace num::pde
