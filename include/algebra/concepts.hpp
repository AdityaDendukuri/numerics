/// @file algebra/concepts.hpp
/// @brief Concepts beside the space and operator concepts: entrywise matrix access and scalar
/// functions.
///
/// The space and operator concepts are in `core/math/concepts.hpp`. Storage layout is
/// described by the `num::repr` predicates in `container/concepts.hpp`.
#pragma once

#include "algebra/scalar.hpp"
#include "core/math/concepts.hpp"
#include "core/types.hpp"
#include <concepts>

namespace num {

/// @brief Linear map presented as an entrywise-indexable 2D array.
///
/// A matrix-free operator is a `linear_operator` but not a `matrix_space`. Routines that read
/// entries, such as pivoting or banded factorization, require this.
template <class A, class T = entry_t<A>>
concept matrix_space = field<T> && requires(const A &a) {
    { a.rows() } -> std::convertible_to<idx>;
    { a.cols() } -> std::convertible_to<idx>;
    { a(idx{0}, idx{0}) } -> std::convertible_to<T>;
};

/// @brief Scalar function \f$f: \mathbb{K} \to \mathbb{K}\f$, any callable.
template <class F, class T = real>
concept scalar_function = field<T> && std::invocable<F, T> && requires(F f, T x) {
    { f(x) } -> std::convertible_to<T>;
};

/// @brief Scalar function supplied with its exact derivative \f$(f, f')\f$, as Newton and
/// Halley's method need.
template <class F, class D, class T = real>
concept differentiable_function = scalar_function<F, T> && scalar_function<D, T>;

} // namespace num
