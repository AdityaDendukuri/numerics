/// @file core/concepts.hpp
/// @brief Storage and operator concepts for numerical routines.
#pragma once

#include "core/types.hpp"
#include "core/vector.hpp"
#include <concepts>
#include <type_traits>

namespace num {

/// @brief Indexed real-valued vector interface.
template<class V>
concept VectorLike = requires(V v, const V cv, idx i) {
  { cv.size() } -> std::convertible_to<idx>;
  { cv[i] } -> std::convertible_to<real>;
  { v[i] } -> std::convertible_to<real>;
};

/// @brief Any object exposing its contiguous real storage as a Vector via
/// .vec(). Satisfied by Vector itself, ScalarField2D, ScalarField3D, ... --
/// lets time integrators and solvers operate on fields without depending on
/// the fields/ module.
template<class T>
concept VecField = requires(T& f) {
  { f.vec() } -> std::same_as<Vector&>;
};

/// @brief Mutable indexed real-valued vector interface.
template<class V>
concept MutableVectorLike = VectorLike<V> && requires(V v, idx i, real a) {
  { v[i] = a } -> std::same_as<real&>;
};

/// @brief Contiguous real-valued vector storage.
template<class V>
concept ContiguousVectorLike = VectorLike<V> && requires(V v, const V cv) {
  { cv.data() } -> std::convertible_to<const real*>;
  { v.data() } -> std::convertible_to<real*>;
};

/// @brief Dense row-major real matrix interface.
template<class A>
concept DenseMatrixLike = requires(A a, const A ca, idx i, idx j) {
  { ca.rows() } -> std::convertible_to<idx>;
  { ca.cols() } -> std::convertible_to<idx>;
  { ca(i, j) } -> std::convertible_to<real>;
  { a(i, j) } -> std::convertible_to<real>;
};

/// @brief Mutable dense row-major real matrix interface.
template<class A>
concept MutableDenseMatrixLike =
  DenseMatrixLike<A> && requires(A a, idx i, idx j, real x) {
    { a(i, j) = x } -> std::same_as<real&>;
  };

/// @brief Contiguous dense row-major real matrix storage.
template<class A>
concept ContiguousDenseMatrixLike = DenseMatrixLike<A> && requires(A a, const A ca) {
  { ca.data() } -> std::convertible_to<const real*>;
  { a.data() } -> std::convertible_to<real*>;
};

/// @brief Compile-time contract for the matrix-free product y = A*x.
template<class Op, class X = Vector, class Y = Vector>
concept LinearOperator =
  VectorLike<X> && MutableVectorLike<Y> && requires(const Op& A, const X& x, Y& y) {
    { A.rows() } -> std::convertible_to<idx>;
    { A.cols() } -> std::convertible_to<idx>;
    { A.apply(x, y) };
  };

/// @brief Operator declared to satisfy \f$A=A^T\f$.
template<class Op, class X = Vector, class Y = Vector>
concept SymmetricLinearOperator = LinearOperator<Op, X, Y> && requires {
  typename std::remove_cvref_t<Op>::symmetric_operator_tag;
};

/// @brief Operator declared to satisfy \f$x^T A x > 0\f$ for all nonzero \f$x\f$.
template<class Op, class X = Vector, class Y = Vector>
concept SPDLinearOperator = SymmetricLinearOperator<Op, X, Y> && requires {
  typename std::remove_cvref_t<Op>::spd_operator_tag;
};

} // namespace num
