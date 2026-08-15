/// @file solvers/preconditioner.hpp
/// @brief Preconditioner concept and diagonal preconditioners.
/// @todo Add SSOR, incomplete Cholesky, ILU(0), and block-Jacobi
/// preconditioners for sparse systems.
#pragma once

#include "core/matrix.hpp"
#include "core/vector.hpp"
#include "linalg/sparse/sparse.hpp"
#include <cmath>
#include <concepts>
#include <stdexcept>
#include <utility>

namespace num {

template<class M>
concept Preconditioner = requires(const M& M_op, const Vector& r, Vector& z) {
  { M_op.rows() } -> std::convertible_to<idx>;
  { M_op.cols() } -> std::convertible_to<idx>;
  M_op.apply(r, z);
};

class JacobiPreconditioner final {
public:
  explicit JacobiPreconditioner(Vector inv_diag)
      : inv_diag_(std::move(inv_diag)) {}

  [[nodiscard]] idx rows() const noexcept { return inv_diag_.size(); }
  [[nodiscard]] idx cols() const noexcept { return inv_diag_.size(); }

  void apply(const Vector& r, Vector& z) const {
    const idx n = inv_diag_.size();
    if (r.size() != n) {
      throw std::invalid_argument("JacobiPreconditioner: dimension mismatch");
    }
    if (z.size() != n) {
      z = Vector(n, 0.0);
    }
    for (idx i = 0; i < n; ++i) {
      z[i] = inv_diag_[i] * r[i];
    }
  }

private:
  Vector inv_diag_;
};

[[nodiscard]] inline JacobiPreconditioner jacobi_preconditioner(const Matrix& A) {
  if (A.rows() != A.cols()) {
    throw std::invalid_argument("jacobi_preconditioner: matrix must be square");
  }
  Vector inv(A.rows());
  for (idx i = 0; i < A.rows(); ++i) {
    if (std::abs(A(i, i)) < real(1e-15)) {
      throw std::invalid_argument("jacobi_preconditioner: zero diagonal");
    }
    inv[i] = real(1) / A(i, i);
  }
  return JacobiPreconditioner(std::move(inv));
}

[[nodiscard]] inline JacobiPreconditioner jacobi_preconditioner(const SparseMatrix& A) {
  if (A.n_rows() != A.n_cols()) {
    throw std::invalid_argument("jacobi_preconditioner: matrix must be square");
  }
  Vector inv(A.n_rows(), 0.0);
  for (idx i = 0; i < A.n_rows(); ++i) {
    const idx row_begin = A.row_ptr()[i];
    const idx row_end = A.row_ptr()[i + 1];
    for (idx p = row_begin; p < row_end; ++p) {
      if (A.col_idx()[p] == i) {
        inv[i] = A.values()[p];
        break;
      }
    }
    if (std::abs(inv[i]) < real(1e-15)) {
      throw std::invalid_argument("jacobi_preconditioner: zero diagonal");
    }
    inv[i] = real(1) / inv[i];
  }
  return JacobiPreconditioner(std::move(inv));
}

} // namespace num
