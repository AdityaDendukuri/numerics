/// @file expv.hpp
/// @brief Krylov subspace matrix exponential-vector product: compute exp(t*A)*v
///
/// Uses the Arnoldi process to build a Krylov basis, then applies a dense
/// Pade [6/6] matrix exponential on the projected (small) problem.
#pragma once

#include "core/types.hpp"
#include "core/vector.hpp"
#include "linalg/solvers/cg.hpp"
#include "linalg/sparse/sparse.hpp"
#include <functional>

namespace num {

/// @brief Compute exp(t*A)*v via Krylov-Pade approximation (matrix-free)
///
/// @param t       Scalar multiplier on A
/// @param matvec  Callable y = A*x
/// @param n       Dimension of the state space (size of v)
/// @param v       Input vector
/// @param m_max   Maximum Krylov dimension (default 30)
/// @param tol     Breakdown tolerance for Arnoldi (default 1e-8)
/// @return        Approximation of exp(t*A)*v
Vector expv(real t, const MatVecFn &matvec, idx n, const Vector &v,
            int m_max = 30, real tol = 1e-8);

/// @brief Compute exp(t*A)*v via Krylov-Pade approximation (sparse matrix)
///
/// @param t    Scalar multiplier on A
/// @param A    Sparse matrix in CSR format
/// @param v    Input vector
/// @param m_max   Maximum Krylov dimension (default 30)
/// @param tol     Breakdown tolerance for Arnoldi (default 1e-8)
/// @return        Approximation of exp(t*A)*v
Vector expv(real t, const SparseMatrix &A, const Vector &v, int m_max = 30,
            real tol = 1e-8);

} // namespace num
