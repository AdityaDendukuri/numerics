#include "linalg/solvers/auto_linear.hpp"
#include "linalg/factorization/lu.hpp"
#include "linalg/sparse/klu.hpp"
#include <optional>
#include <stdexcept>

namespace num {

struct AutoLinearSolver::Impl {
  idx n = 0;
  std::optional<LUResult> dense_factor;
  std::unique_ptr<KLUFactor> sparse_factor;
};

AutoLinearSolver::AutoLinearSolver(const SparseMatrix& matrix, AutoLinearOptions options)
    : impl_(std::make_unique<Impl>()) {
  if (matrix.n_rows() != matrix.n_cols()) {
    throw std::invalid_argument("AutoLinearSolver requires a square matrix");
  }
  impl_->n = matrix.n_rows();
  if (matrix.n_rows() > options.dense_limit && klu_available()) {
    impl_->sparse_factor = std::make_unique<KLUFactor>(matrix);
  } else {
    impl_->dense_factor = lu(dense(matrix));
    if (impl_->dense_factor->singular) {
      throw std::runtime_error("AutoLinearSolver encountered a singular matrix");
    }
  }
}

AutoLinearSolver::~AutoLinearSolver() = default;
AutoLinearSolver::AutoLinearSolver(AutoLinearSolver&&) noexcept = default;
AutoLinearSolver& AutoLinearSolver::operator=(AutoLinearSolver&&) noexcept = default;

idx AutoLinearSolver::size() const noexcept {
  return impl_ ? impl_->n : 0;
}

void AutoLinearSolver::solve(const Vector& rhs, Vector& solution) const {
  if (impl_->sparse_factor) {
    impl_->sparse_factor->solve(rhs, solution);
  } else {
    lu_solve(*impl_->dense_factor, rhs, solution);
  }
}

void AutoLinearSolver::solve(const Matrix& rhs, Matrix& solution) const {
  if (impl_->sparse_factor) {
    impl_->sparse_factor->solve(rhs, solution);
  } else {
    lu_solve(*impl_->dense_factor, rhs, solution);
  }
}

void AutoLinearSolver::solve_transpose(const Vector& rhs, Vector& solution) const {
  if (impl_->sparse_factor) {
    impl_->sparse_factor->solve_transpose(rhs, solution);
  } else {
    lu_solve_transpose(*impl_->dense_factor, rhs, solution);
  }
}

void AutoLinearSolver::solve_transpose(const Matrix& rhs, Matrix& solution) const {
  if (impl_->sparse_factor) {
    impl_->sparse_factor->solve_transpose(rhs, solution);
  } else {
    lu_solve_transpose(*impl_->dense_factor, rhs, solution);
  }
}

void AutoLinearSolver::solve_in_place(Vector& right_hand_side) const {
  Vector solution(right_hand_side.size(), 0.0);
  solve(right_hand_side, solution);
  right_hand_side = std::move(solution);
}

void AutoLinearSolver::solve_in_place(Matrix& right_hand_sides) const {
  Matrix solution;
  solve(right_hand_sides, solution);
  right_hand_sides = std::move(solution);
}

} // namespace num
