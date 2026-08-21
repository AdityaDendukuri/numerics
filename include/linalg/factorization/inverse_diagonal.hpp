/// @file linalg/factorization/inverse_diagonal.hpp
/// @brief Diagonal of an inverse from reusable factorizations.
#pragma once

#include "linalg/factorization/cholesky.hpp"
#include "linalg/factorization/lu.hpp"
#include "linalg/solvers/auto_linear.hpp"
#include "linalg/sparse/klu.hpp"
#include <span>

namespace num {

/// Reusable dense buffers for blocked and selected inverse extraction.
struct InverseDiagonalWorkspace {
  Matrix right_hand_sides; ///< Reused block of selected identity columns.
  Matrix solutions; ///< Reused solution block.
};

/// Algorithm used to obtain an updated inverse diagonal.
enum class InverseDiagonalUpdatePath {
  woodbury,
  direct,
};

/// Compute diag(A^-1) through blocked identity solves.
void inverse_diagonal(const LUResult& factor,
                      Vector& result,
                      InverseDiagonalWorkspace& workspace,
                      idx block_size = 64);
/// Compute diag(A^-1) from a Cholesky factor using blocked identity solves.
void inverse_diagonal(const CholeskyResult& factor,
                      Vector& result,
                      InverseDiagonalWorkspace& workspace,
                      idx block_size = 64);
/// Compute diag(A^-1) from a KLU factor using blocked identity solves.
void inverse_diagonal(const KLUFactor& factor,
                      Vector& result,
                      InverseDiagonalWorkspace& workspace,
                      idx block_size = 64);
/// Compute diag(A^-1) from an automatically selected reusable factor.
void inverse_diagonal(const AutoLinearSolver& factor,
                      Vector& result,
                      InverseDiagonalWorkspace& workspace,
                      idx block_size = 64);

/// Compute A^-1(rows[i], columns[i]) using only the requested inverse columns.
void selected_inverse(const LUResult& factor,
                      std::span<const idx> rows,
                      std::span<const idx> columns,
                      Vector& result,
                      InverseDiagonalWorkspace& workspace);
/// Compute selected inverse entries from a Cholesky factor.
void selected_inverse(const CholeskyResult& factor,
                      std::span<const idx> rows,
                      std::span<const idx> columns,
                      Vector& result,
                      InverseDiagonalWorkspace& workspace);
/// Compute selected inverse entries from a KLU factor.
void selected_inverse(const KLUFactor& factor,
                      std::span<const idx> rows,
                      std::span<const idx> columns,
                      Vector& result,
                      InverseDiagonalWorkspace& workspace);
/// Compute selected inverse entries from an automatically selected factor.
void selected_inverse(const AutoLinearSolver& factor,
                      std::span<const idx> rows,
                      std::span<const idx> columns,
                      Vector& result,
                      InverseDiagonalWorkspace& workspace);

/// Extract A^-1(indices, indices), preserving the requested index order.
void inverse_principal_block(const LUResult& factor,
                             std::span<const idx> indices,
                             Matrix& result,
                             InverseDiagonalWorkspace& workspace);
/// Extract a principal inverse block from a Cholesky factor.
void inverse_principal_block(const CholeskyResult& factor,
                             std::span<const idx> indices,
                             Matrix& result,
                             InverseDiagonalWorkspace& workspace);
/// Extract a principal inverse block from a KLU factor.
void inverse_principal_block(const KLUFactor& factor,
                             std::span<const idx> indices,
                             Matrix& result,
                             InverseDiagonalWorkspace& workspace);
/// Extract a principal inverse block from an automatically selected factor.
void inverse_principal_block(const AutoLinearSolver& factor,
                             std::span<const idx> indices,
                             Matrix& result,
                             InverseDiagonalWorkspace& workspace);

/// Update diag(A^-1) after A <- A + U W U^T, with direct solves as fallback.
///
/// `factor` belongs to A and `updated_factor` to the updated matrix. The
/// Cholesky overload uses the symmetric squared-norm formula when W is SPD;
/// other overloads use the two-sided formula needed for nonsymmetric A.
[[nodiscard]] InverseDiagonalUpdatePath inverse_diagonal_after_update(
  const LUResult& factor,
  const LUResult& updated_factor,
  const Vector& diagonal,
  const Matrix& update_vectors,
  const Matrix& update_weights,
  Vector& result,
  InverseDiagonalWorkspace& workspace,
  idx fallback_block_size = 64);
/// Apply the symmetric Woodbury diagonal formula to Cholesky factors.
[[nodiscard]] InverseDiagonalUpdatePath inverse_diagonal_after_update(
  const CholeskyResult& factor,
  const CholeskyResult& updated_factor,
  const Vector& diagonal,
  const Matrix& update_vectors,
  const Matrix& update_weights,
  Vector& result,
  InverseDiagonalWorkspace& workspace,
  idx fallback_block_size = 64);
/// Apply the two-sided Woodbury diagonal formula to KLU factors.
[[nodiscard]] InverseDiagonalUpdatePath inverse_diagonal_after_update(
  const KLUFactor& factor,
  const KLUFactor& updated_factor,
  const Vector& diagonal,
  const Matrix& update_vectors,
  const Matrix& update_weights,
  Vector& result,
  InverseDiagonalWorkspace& workspace,
  idx fallback_block_size = 64);
/// Apply the two-sided Woodbury formula through an automatic factor backend.
[[nodiscard]] InverseDiagonalUpdatePath inverse_diagonal_after_update(
  const AutoLinearSolver& factor,
  const AutoLinearSolver& updated_factor,
  const Vector& diagonal,
  const Matrix& update_vectors,
  const Matrix& update_weights,
  Vector& result,
  InverseDiagonalWorkspace& workspace,
  idx fallback_block_size = 64);

} // namespace num
