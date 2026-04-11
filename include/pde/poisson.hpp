/// @file pde/poisson.hpp
/// @brief 2D Poisson equation solved via the Discrete Sine Transform.
///
/// Solves  -Delta u = f  on (0,1)^2 with homogeneous Dirichlet BC,
/// discretised on an N x N interior grid with h = 1/(N+1).
/// N must equal 2^p - 1 so that the odd-extension length 2(N+1) is a
/// power of two (required by the radix-2 FFT backend).
///
/// Algorithm (Demmel, CS267 Lecture 20):
///   L_2 = L_1 x I + I x L_1,  L_1 = tridiag(-1, 2, -1).
///   L_1 = F * D * F^T,  F_{jk} = sin(j*k*pi/(N+1)).
///   Transformed system:  u_hat_{jk} = h^2 * f_hat_{jk} / (D_j + D_k).
///   DST-I computed via complex FFT on an odd-extended sequence.
///   Cost: O(N^2 log N).
///
/// Two variants:
///   poisson2d_fd   -- FD eigenvalues D_k = 2(1 - cos(k*pi/(N+1))).  Error O(h^2).
///   poisson2d      -- Exact eigenvalues (k*pi)^2.  Machine-precision error for
///                     f in the DST eigenbasis; exponential convergence otherwise.
///
/// Reference: J. Demmel, CS267 Lecture 20, UC Berkeley, 2025.
#pragma once

#include "core/matrix.hpp"

namespace num {
namespace pde {

/// @brief Solve -Delta u = f on (0,1)^2 (Dirichlet) via DST with FD eigenvalues.
///
/// @param f  N x N matrix of RHS values on the interior grid (row i, col j
///           corresponds to the grid point ((i+1)*h, (j+1)*h)).
/// @param N  Grid dimension.  Must satisfy N = 2^p - 1.
/// @return   N x N solution matrix u.
[[nodiscard]] Matrix poisson2d_fd(const Matrix& f, int N);

/// @brief Solve -Delta u = f on (0,1)^2 (Dirichlet) via DST with exact eigenvalues.
///
/// Replaces the FD eigenvalue D_k = 2(1 - cos(k*pi/(N+1))) with the exact
/// eigenvalue (k*pi)^2 of the continuous operator -d^2/dx^2.  The error is
/// determined by the quadrature approximation of the DST projection rather
/// than the FD truncation of the eigenvalue, giving spectral accuracy.
///
/// @param f  N x N matrix of RHS values.
/// @param N  Grid dimension.  Must satisfy N = 2^p - 1.
/// @return   N x N solution matrix u.
[[nodiscard]] Matrix poisson2d(const Matrix& f, int N);

} // namespace pde
} // namespace num
