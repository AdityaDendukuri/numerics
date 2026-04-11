/// @file eigen/lanczos.hpp
/// @brief Lanczos algorithm  -- k extreme eigenvalues of a large sparse matrix
///
/// The Lanczos algorithm builds a k-dimensional Krylov subspace
///
///   K_k(A, v) = span{v, Av, A^2v, ..., A^{k-1}v}
///
/// and finds the k Ritz pairs (approximate eigenpairs) that are the extreme
/// eigenvalues of the projected kxk tridiagonal matrix T_k.
///
/// Key properties:
///   - Matrix-free: A is given as a MatVecFn callable (works with sparse,
///   structured A)
///   - O(k*nnz) work for k steps (nnz = cost of one matvec)
///   - Finds the k LARGEST and k SMALLEST eigenvalues accurately
///
/// Usage:
/// @code
///   lanczos(mv, n, k);                          // sequential
///   lanczos(mv, n, k, 1e-10, 0, num::omp);     // OMP reorthogonalisation
/// @endcode
#pragma once

#include "core/matrix.hpp"
#include "core/policy.hpp"
#include "core/vector.hpp"
#include "linalg/eigen/jacobi_eig.hpp"
#include "linalg/solvers/cg.hpp" // MatVecFn

namespace num {

/// @brief Result of a Lanczos eigensolver
struct LanczosResult {
  Vector ritz_values;  ///< k Ritz values (approximate eigenvalues), ascending
  Matrix ritz_vectors; ///< nxk matrix  -- each column is a Ritz vector
  idx steps = 0;       ///< Actual Lanczos steps taken
  bool converged =
      false; ///< Whether all requested Ritz pairs met the tolerance
};

/// @brief Lanczos eigensolver for large sparse symmetric matrices.
///
/// @param matvec    Callable computing w = A*v (matrix-free)
/// @param n         Dimension of the problem
/// @param k         Number of eigenvalues requested (k << n)
/// @param tol       Convergence on ||A*u - lambda*u|| for each Ritz pair
/// @param max_steps Maximum Lanczos steps (default: min(3k, n))
/// @param backend   Backend for reorthogonalisation inner products (default:
/// seq)
LanczosResult lanczos(MatVecFn matvec, idx n, idx k, real tol = 1e-10,
                      idx max_steps = 0, Backend backend = Backend::seq);

} // namespace num
