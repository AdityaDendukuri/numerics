/// @file linalg/matrix_properties.hpp
/// @brief Declared mathematical properties for stored matrices.
#pragma once

#include "core/matrix.hpp"
#include <cmath>
#include <stdexcept>
#include <utility>

namespace num::linalg {

[[nodiscard]] inline bool is_symmetric(const Matrix& A, real tol = 1e-12) {
  if (A.rows() != A.cols()) {
    return false;
  }
  const idx n = A.rows();
  for (idx i = 0; i < n; ++i) {
    for (idx j = 0; j < i; ++j) {
      if (std::abs(A(i, j) - A(j, i)) > tol) {
        return false;
      }
    }
  }
  return true;
}

[[nodiscard]] inline bool is_spd(const Matrix& A, real tol = 1e-12) {
  if (!is_symmetric(A, tol)) {
    return false;
  }

  const idx n = A.rows();
  Matrix L(n, n, 0.0);
  for (idx i = 0; i < n; ++i) {
    for (idx j = 0; j <= i; ++j) {
      real sum = A(i, j);
      for (idx k = 0; k < j; ++k) {
        sum -= L(i, k) * L(j, k);
      }

      if (i == j) {
        if (sum <= tol) {
          return false;
        }
        L(i, j) = std::sqrt(sum);
      } else {
        L(i, j) = sum / L(j, j);
      }
    }
  }
  return true;
}

template<class Mat>
class SymmetricMatrix final {
public:
  using symmetric_matrix_tag = void;

  explicit SymmetricMatrix(Mat A)
      : A_(std::move(A)) {}

  [[nodiscard]] const Mat& base() const noexcept { return A_; }
  [[nodiscard]] idx rows() const noexcept { return A_.rows(); }
  [[nodiscard]] idx cols() const noexcept { return A_.cols(); }

private:
  Mat A_;
};

template<class Mat>
class SPDMatrix final {
public:
  using symmetric_matrix_tag = void;
  using spd_matrix_tag = void;

  explicit SPDMatrix(Mat A)
      : A_(std::move(A)) {}

  [[nodiscard]] const Mat& base() const noexcept { return A_; }
  [[nodiscard]] idx rows() const noexcept { return A_.rows(); }
  [[nodiscard]] idx cols() const noexcept { return A_.cols(); }

private:
  Mat A_;
};

[[nodiscard]] inline SymmetricMatrix<Matrix> assume_symmetric(Matrix A) {
  return SymmetricMatrix<Matrix>(std::move(A));
}

[[nodiscard]] inline SPDMatrix<Matrix> assume_spd(Matrix A) {
  return SPDMatrix<Matrix>(std::move(A));
}

[[nodiscard]] inline SymmetricMatrix<Matrix> make_symmetric(Matrix A, real tol = 1e-12) {
  if (!is_symmetric(A, tol)) {
    throw std::invalid_argument("make_symmetric: matrix is not symmetric");
  }
  return SymmetricMatrix<Matrix>(std::move(A));
}

[[nodiscard]] inline SPDMatrix<Matrix> make_spd(Matrix A, real tol = 1e-12) {
  if (!is_spd(A, tol)) {
    throw std::invalid_argument("make_spd: matrix is not symmetric positive definite");
  }
  return SPDMatrix<Matrix>(std::move(A));
}

} // namespace num::linalg
