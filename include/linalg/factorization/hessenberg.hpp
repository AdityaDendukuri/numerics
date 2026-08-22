/// @file linalg/factorization/hessenberg.hpp
/// @brief Upper Hessenberg decomposition A = Q H Q^T via Householder reflections.
#pragma once

#include "core/debug.hpp"
#include "core/matrix.hpp"
#include "core/types.hpp"
#include "core/vector.hpp"

namespace num {

/// @brief Upper Hessenberg decomposition of a square matrix: A = Q H Q^T.
class HessenbergDecomposition {
  public:
    /// Compute the Hessenberg decomposition of square matrix A.
    explicit HessenbergDecomposition(const Matrix &A);

    [[nodiscard]] idx size() const noexcept { return H_.rows(); }
    [[nodiscard]] const Matrix &H() const noexcept { return H_; }
    [[nodiscard]] const Matrix &Q() const noexcept { return Q_; }

  private:
    Matrix H_;
    Matrix Q_;
};

/// Compute the upper Hessenberg decomposition of a square matrix.
[[nodiscard]] inline HessenbergDecomposition hessenberg(const Matrix &A) {
    return HessenbergDecomposition(A);
}

} // namespace num
