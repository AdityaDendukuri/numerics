/// @file operator/concepts.hpp
/// @brief Compile-time contracts for matrix-free linear operators.
#pragma once

#include "core/concepts.hpp"
#include <type_traits>

namespace num {

/// Object supporting the matrix-free product y=A*x.
template <class Op, class X = Vector, class Y = Vector>
concept LinearOperator =
    VectorLike<X> && MutableVectorLike<Y> && requires(const Op &A, const X &x, Y &y) {
    {A.rows()}->std::convertible_to<idx>;
    {A.cols()}->std::convertible_to<idx>;
    {A.apply(x, y)};
};

/// Linear operator carrying a compile-time symmetry guarantee.
template <class Op, class X = Vector, class Y = Vector>
concept SymmetricLinearOperator = LinearOperator<Op, X, Y> && requires {
    typename std::remove_cvref_t<Op>::symmetric_operator_tag;
};

/// Symmetric operator carrying a compile-time positive-definiteness guarantee.
template <class Op, class X = Vector, class Y = Vector>
concept SPDLinearOperator = SymmetricLinearOperator<Op, X, Y> && requires {
    typename std::remove_cvref_t<Op>::spd_operator_tag;
};

} // namespace num
