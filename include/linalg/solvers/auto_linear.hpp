/// @file linalg/solvers/auto_linear.hpp
/// @brief Automatic dense/sparse factorization for reusable real solves.
#pragma once

#include "core/matrix.hpp"
#include "core/vector.hpp"
#include "linalg/sparse/sparse.hpp"
#include <memory>

namespace num {

/// Select dense LU at or below `dense_limit`, otherwise prefer SuiteSparse KLU.
struct AutoLinearOptions {
  idx dense_limit = 32;
};

/// Reusable real factorization that selects a dense or sparse backend by size.
class AutoLinearSolver {
public:
  /// Factor a square CSR matrix using the configured backend threshold.
  explicit AutoLinearSolver(const SparseMatrix& matrix, AutoLinearOptions options = {});
  ~AutoLinearSolver();
  AutoLinearSolver(AutoLinearSolver&&) noexcept;
  AutoLinearSolver& operator=(AutoLinearSolver&&) noexcept;
  AutoLinearSolver(const AutoLinearSolver&) = delete;
  AutoLinearSolver& operator=(const AutoLinearSolver&) = delete;

  /// Return the order of the factored matrix, or zero after a move.
  [[nodiscard]] idx size() const noexcept;
  /// Solve AX=B without modifying the stored factorization.
  void solve(const Vector& rhs, Vector& solution) const;
  void solve(const Matrix& rhs, Matrix& solution) const;
  /// Solve A^T x=b without modifying the stored factorization.
  void solve_transpose(const Vector& rhs, Vector& solution) const;
  /// Solve A^T X=B for several dense right-hand sides.
  void solve_transpose(const Matrix& rhs, Matrix& solution) const;
  /// Replace one or more right-hand sides with their solutions.
  void solve_in_place(Vector& right_hand_side) const;
  void solve_in_place(Matrix& right_hand_sides) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace num
