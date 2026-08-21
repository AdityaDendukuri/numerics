/// @file operator/dense.hpp
/// @brief Dense Matrix adapter for the operator protocol.
#pragma once

#include "core/matrix.hpp"
#include "core/policy.hpp"
#include "operator/concepts.hpp"

namespace num::operators {

/// @brief Adapt a dense Matrix to the operator protocol.
struct DenseOp final {
  /// Store a non-owning matrix reference and backend selection.
  explicit DenseOp(const Matrix& A, Backend b = default_backend)
      : A_(A),
        b_(b) {}

  /// Compute y=A*x using the selected backend.
  void apply(const Vector& x, Vector& y) const;
  [[nodiscard]] idx rows() const noexcept { return A_.rows(); }
  [[nodiscard]] idx cols() const noexcept { return A_.cols(); }

private:
  const Matrix& A_;
  Backend b_;
};

static_assert(LinearOperator<DenseOp>);

} // namespace num::operators
