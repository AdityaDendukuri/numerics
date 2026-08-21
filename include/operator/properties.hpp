/// @file operator/properties.hpp
/// @brief Declared mathematical properties for linear operators.
/// @todo Add optional sampled diagnostics for matrix-free symmetry and
/// positive-definiteness checks.
#pragma once

#include "core/concepts.hpp"
#include "core/debug.hpp"
#include <utility>

namespace num::operators {

template<class Op>
  requires LinearOperator<Op>
class SymmetricOp final {
public:
  using symmetric_operator_tag = void;

  explicit SymmetricOp(Op op)
      : op_(std::move(op)) {}

  void apply(const Vector& x, Vector& y) const { op_.apply(x, y); }
  [[nodiscard]] idx rows() const noexcept { return op_.rows(); }
  [[nodiscard]] idx cols() const noexcept { return op_.cols(); }

  [[nodiscard]] const Op& base() const noexcept { return op_; }

private:
  Op op_;
};

template<class Op>
  requires LinearOperator<Op>
class SPDOp final {
public:
  using symmetric_operator_tag = void;
  using spd_operator_tag = void;

  explicit SPDOp(Op op)
      : op_(std::move(op)) {}

  void apply(const Vector& x, Vector& y) const { op_.apply(x, y); }
  [[nodiscard]] idx rows() const noexcept { return op_.rows(); }
  [[nodiscard]] idx cols() const noexcept { return op_.cols(); }

  [[nodiscard]] const Op& base() const noexcept { return op_; }

private:
  Op op_;
};

template<class Op>
  requires LinearOperator<Op>
[[nodiscard]] SymmetricOp<Op> assume_symmetric(Op op) {
  debug::verify_symmetry_sample<Op, Vector>(op, op.cols());
  return SymmetricOp<Op>(std::move(op));
}

template<class Op>
  requires LinearOperator<Op>
[[nodiscard]] SPDOp<Op> assume_spd(Op op) {
  debug::verify_spd_sample<Op, Vector>(op, op.cols());
  return SPDOp<Op>(std::move(op));
}

} // namespace num::operators


