/// @file lu.hpp
/// @brief LU factorization with partial pivoting
#pragma once

#include "core/matrix.hpp"
#include "core/policy.hpp"
#include <vector>

namespace num {

/// @brief Result of an LU factorization with partial pivoting (PA = LU)
///
/// L and U are stored packed in a single matrix:
///   - U occupies the upper triangle (including the diagonal)
///   - L occupies the strict lower triangle (diagonal of L is implicitly 1)
///
/// piv[k] = the row index that was swapped into position k at step k.
/// singular is set if any diagonal of U is below the tolerance 1e-14.
struct LUResult {
  Matrix LU;
  std::vector<idx> piv;
  bool singular = false;
};

/// @brief LU factorization of a square matrix A with partial pivoting.
///
/// Computes P, L, U such that P*A = L*U where:
///   P  = permutation matrix encoded in piv
///   L  = unit lower triangular  (diagonal = 1, stored below diag of LU)
///   U  = upper triangular       (stored on and above diag of LU)
///
/// @param backend  Backend::lapack uses LAPACKE_dgetrf (default when
/// available).
///                 Backend::seq    uses our Doolittle implementation.
LUResult lu(const Matrix &A, Backend backend = lapack_backend);

/// @brief Solve A*x = b using a precomputed LU factorization.
///
/// Three steps:
///   1. Apply permutation:     y = P*b
///   2. Forward substitution:  solve L*z = y
///   3. Backward substitution: solve U*x = z
void lu_solve(const LUResult &f, const Vector &b, Vector &x);

/// @brief Solve A*X = B for multiple right-hand sides.
///
/// Each column of B is solved independently using the same factorization.
/// B and X are nxnrhs matrices (column-major access pattern).
void lu_solve(const LUResult &f, const Matrix &B, Matrix &X);

/// @brief Determinant of A from its LU factorization.
///
/// det(A) = det(P)^{-1} * prod(U[i,i])
///        = (-1)^{swaps} * prod(U[i,i])
///
/// Overflow is possible for large n  -- use log-determinant for stability.
real lu_det(const LUResult &f);

/// @brief Inverse of A from its LU factorization.
///
/// Solves A * X = I column by column.  O(n^3)  -- only use when the full
/// inverse is genuinely needed (prefer lu_solve for specific right-hand sides).
Matrix lu_inv(const LUResult &f);

} // namespace num
