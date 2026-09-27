/// @file concepts.hpp
/// @brief The concepts: scalars, spaces, operators, and operators claiming a law.
///
/// ```
/// field<T>                         floating point, real or complex
/// vector_space<V>                  dimension, zero_like, scale, axpy over a field
///  └ inner_product_space<V>        + inner, norm
/// linear_operator<Op, X, Y>        apply, rows, cols between vector spaces
///  └ self_adjoint_operator<Op, V>  + claims law::self_adjoint
///     └ psd_operator<Op, V>        + claims law::psd
///        └ spd_operator<Op, V>     + claims law::spd
/// ```
///
/// Spaces are decided by their operations alone. Operator laws are declared, since the
/// compiler cannot decide them; see `core/math/laws.hpp`. The header depends only on the
/// standard library.
#pragma once

#include "core/math/associated.hpp"
#include "core/math/laws.hpp"
#include "core/math/operations.hpp"
#include <concepts>
#include <type_traits>

namespace num::math {

/// @brief Scalar field \f$\mathbb{K}\f$: a floating-point type, or `std::complex` of one.
template <class T>
concept field = std::floating_point<T> ||
                (scalars::is_complex_v<T> && std::floating_point<scalars::real_t<T>>);

/// @brief Vector space over a field: sized, copyable, zeroable, and closed under `scale` and
/// `axpy`.
template <class V>
concept vector_space = field<scalar_t<V>> && std::copy_constructible<V> &&
                       requires(V &v, const V &x, scalar_t<V> a) {
    { dimension(x) } -> std::integral;
    { zero_like(x) } -> std::same_as<V>;
    scale(a, v);
    axpy(a, x, v);
};

/// @brief Vector space with an inner product and its induced norm, as Krylov methods need.
template <class V>
concept inner_product_space = vector_space<V> && requires(const V &x, const V &y) {
    { inner(x, y) } -> std::convertible_to<scalar_t<V>>;
    norm(x);
};

/// @brief Linear map \f$A: X \to Y\f$ that can be applied and knows its shape.
///
/// Linearity is a precondition the caller meets, like the semantic requirements of the
/// standard concepts; it has no law, since no algorithm is chosen by it.
template <class Op, class X = domain_t<Op>, class Y = codomain_t<Op>>
concept linear_operator = vector_space<X> && vector_space<Y> &&
                          requires(const Op &A, const X &x, Y &y) {
    apply(A, x, y);
    { A.rows() } -> std::integral;
    { A.cols() } -> std::integral;
};

/// @brief \f$A = A^*\f$ on V: real spectrum. The precondition for MINRES and Lanczos.
template <class Op, class V = domain_t<Op>>
concept self_adjoint_operator = linear_operator<Op, V, V> && claims<Op, law::self_adjoint>;

/// @brief \f$\langle x, Ax \rangle \ge 0\f$ on V: graph Laplacians and Gram matrices.
template <class Op, class V = domain_t<Op>>
concept psd_operator = self_adjoint_operator<Op, V> && claims<Op, law::psd>;

/// @brief \f$\langle x, Ax \rangle > 0\f$ on V: invertible. The precondition for CG.
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
