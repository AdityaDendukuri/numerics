/// @file operator/concepts.hpp
/// @brief Compile-time contracts for matrix-free linear and nonlinear operators.
#pragma once

#include "core/concepts.hpp"
#include <type_traits>

namespace num {

/// @brief Object supporting the matrix-free product y=A*x.
template <class Op, class X = Vector, class Y = Vector>
concept LinearOperator =
    VectorLike<X> && MutableVectorLike<Y> && requires(const Op &A, const X &x, Y &y) {
    { A.rows() } -> std::convertible_to<idx>;
    { A.cols() } -> std::convertible_to<idx>;
    { A.apply(x, y) };
};

/// @brief Linear operator supporting both forward y=A*x and adjoint x=A^*y products.
template <class Op, class X = Vector, class Y = Vector>
concept AdjointableLinearOperator =
    LinearOperator<Op, X, Y> && requires(const Op &A, const Y &y, X &x) {
    { A.apply_adjoint(y, x) };
};

/// @brief Linear operator carrying a compile-time symmetry guarantee.
template <class Op, class X = Vector, class Y = Vector>
concept SymmetricLinearOperator = LinearOperator<Op, X, Y> && requires {
    typename std::remove_cvref_t<Op>::symmetric_operator_tag;
};

/// @brief Symmetric operator carrying a compile-time positive-definiteness guarantee.
template <class Op, class X = Vector, class Y = Vector>
concept SPDLinearOperator = SymmetricLinearOperator<Op, X, Y> && requires {
    typename std::remove_cvref_t<Op>::spd_operator_tag;
};

/// @brief Complex or real self-adjoint operator contract.
template <class Op, class X = Vector, class Y = Vector>
concept HermitianLinearOperator = SymmetricLinearOperator<Op, X, Y>;

/// @brief Nonlinear mapping F(x) = y between finite-dimensional vector spaces.
template <class Op, class X = Vector, class Y = Vector>
concept NonlinearOperator =
    VectorLike<X> && MutableVectorLike<Y> && requires(const Op &F, const X &x, Y &y) {
    { F.rows() } -> std::convertible_to<idx>;
    { F.cols() } -> std::convertible_to<idx>;
    { F.apply(x, y) };
};

} // namespace num
