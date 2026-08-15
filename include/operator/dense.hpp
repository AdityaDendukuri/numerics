/// @file operator/dense.hpp
/// @brief Dense Matrix adapter for the operator protocol.
#pragma once

#include "core/concepts.hpp"
#include "core/matrix.hpp"
#include "core/policy.hpp"

namespace num::operators {

/// @brief Adapt a dense Matrix to the operator protocol.
struct DenseOp final {
  explicit DenseOp(const Matrix& A, Backend b = default_backend)
      : A_(A),
        b_(b) {}

  void apply(const Vector& x, Vector& y) const;
  [[nodiscard]] idx rows() const noexcept { return A_.rows(); }
  [[nodiscard]] idx cols() const noexcept { return A_.cols(); }

private:
  const Matrix& A_;
  Backend b_;
};

static_assert(LinearOperator<DenseOp>);

} // namespace num::operators
