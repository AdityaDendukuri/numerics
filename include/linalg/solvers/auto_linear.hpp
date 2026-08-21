/// @file linalg/solvers/auto_linear.hpp
/// @brief Automatic dense/sparse factorization for reusable real solves.
#pragma once

#include "core/matrix.hpp"
#include "core/vector.hpp"
#include "linalg/sparse/sparse.hpp"
#include <memory>

namespace num {

struct AutoLinearOptions {
  idx dense_limit = 32;
};

class AutoLinearSolver {
public:
  explicit AutoLinearSolver(const SparseMatrix& matrix, AutoLinearOptions options = {});
  ~AutoLinearSolver();
  AutoLinearSolver(AutoLinearSolver&&) noexcept;
  AutoLinearSolver& operator=(AutoLinearSolver&&) noexcept;
  AutoLinearSolver(const AutoLinearSolver&) = delete;
  AutoLinearSolver& operator=(const AutoLinearSolver&) = delete;

  [[nodiscard]] idx size() const noexcept;
  void solve(const Vector& rhs, Vector& solution) const;
  void solve(const Matrix& rhs, Matrix& solution) const;
  void solve_transpose(const Vector& rhs, Vector& solution) const;
  void solve_in_place(Vector& right_hand_side) const;
  void solve_in_place(Matrix& right_hand_sides) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace num
