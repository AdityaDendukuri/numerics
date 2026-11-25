/// @file core/math/concepts.hpp
/// @brief Concepts for scalars, vector spaces and linear operators.
#pragma once

#include "core/math/associated.hpp"
#include "core/math/laws.hpp"
#include "core/math/operations.hpp"
#include <concepts>
#include <type_traits>

namespace num::math {

/// @brief Scalar field \f$\mathbb{K}\f$: a floating-point type or `std::complex` of one.
template <class T>
concept field = std::floating_point<T> ||
                (scalars::is_complex_v<T> && std::floating_point<scalars::real_t<T>>);

/// @brief Vector space over \f$\mathbb{K}\f$: sized, zeroable, closed under `scale` and `axpy`.
template <class V>
concept vector_space = field<scalar_t<V>> && std::copy_constructible<V> &&
                       requires(V &v, const V &x, scalar_t<V> a) {
    { dimension(x) } -> std::integral;
    { zero_like(x) } -> std::same_as<V>;
    scale(a, v);
    axpy(a, x, v);
};

/// @brief Inner product space: a vector space with \f$\langle x, y \rangle\f$ and \f$\|x\|\f$.
template <class V>
concept inner_product_space = vector_space<V> && requires(const V &x, const V &y) {
    { inner(x, y) } -> std::convertible_to<scalar_t<V>>;
    norm(x);
};

/// @brief Linear operator \f$A: X \to Y\f$ with `apply`, `rows` and `cols`.
/// Linearity is assumed, not checked.
template <class Op, class X = domain_t<Op>, class Y = codomain_t<Op>>
concept linear_operator = vector_space<X> && vector_space<Y> &&
                          requires(const Op &A, const X &x, Y &y) {
    apply(A, x, y);
    { A.rows() } -> std::integral;
    { A.cols() } -> std::integral;
};

/// @brief Self-adjoint operator: \f$A = A^*\f$.
template <class Op, class V = domain_t<Op>>
concept self_adjoint_operator = linear_operator<Op, V, V> && claims<Op, law::self_adjoint>;

/// @brief Positive semidefinite operator: \f$\langle x, A x \rangle \ge 0\f$.
template <class Op, class V = domain_t<Op>>
concept psd_operator = self_adjoint_operator<Op, V> && claims<Op, law::psd>;

/// @brief Positive definite operator: \f$\langle x, A x \rangle > 0\f$ for \f$x \ne 0\f$.
template <class Op, class V = domain_t<Op>>
concept spd_operator = psd_operator<Op, V> && claims<Op, law::spd>;

} // namespace num::math

namespace num {

using math::field;
using math::inner_product_space;
using math::linear_operator;
using math::psd_operator;
using math::self_adjoint_operator;
using math::spd_operator;
using math::vector_space;

} // namespace num
