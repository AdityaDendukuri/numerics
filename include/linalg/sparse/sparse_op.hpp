/// @file linalg/sparse/sparse_op.hpp
/// @brief SparseMatrix adapter for the operator protocol.
///
/// Lives beside SparseMatrix (rather than under operator/) so the operator
/// module stays free of any linalg dependency; the LinearOperator contract it
/// models is defined in operator/concepts.hpp.
#pragma once

#include "linalg/sparse/sparse.hpp"
#include "operator/concepts.hpp"

namespace num::operators {

/// @brief Adapt a SparseMatrix to the operator protocol.
struct SparseOp final {
  /// Store a non-owning reference to a CSR matrix.
  explicit SparseOp(const SparseMatrix& A)
      : A_(A) {}

  /// Compute y=A*x.
  void apply(const Vector& x, Vector& y) const;
  [[nodiscard]] idx rows() const noexcept { return A_.n_rows(); }
  [[nodiscard]] idx cols() const noexcept { return A_.n_cols(); }

private:
  const SparseMatrix& A_;
};

static_assert(LinearOperator<SparseOp>);

} // namespace num::operators
