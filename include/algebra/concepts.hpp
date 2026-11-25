/// @file algebra/concepts.hpp
/// @brief Concepts for entrywise matrices and scalar functions.
#pragma once

#include "algebra/scalar.hpp"
#include "core/math/concepts.hpp"
#include "core/types.hpp"
#include <concepts>

namespace num {

/// @brief Matrix space: a linear map with entrywise access \f$a_{ij}\f$.
template <class A, class T = entry_t<A>>
concept matrix_space = field<T> && requires(const A &a) {
    { a.rows() } -> std::convertible_to<idx>;
    { a.cols() } -> std::convertible_to<idx>;
    { a(idx{0}, idx{0}) } -> std::convertible_to<T>;
};

/// @brief Scalar function: \f$f : \mathbb{K} \to \mathbb{K}\f$.
template <class F, class T = real>
concept scalar_function = field<T> && std::invocable<F, T> && requires(F f, T x) {
    { f(x) } -> std::convertible_to<T>;
};

/// @brief Differentiable scalar function: \f$f\f$ with its derivative \f$f'\f$.
template <class F, class D, class T = real>
concept differentiable_function = scalar_function<F, T> && scalar_function<D, T>;

} // namespace num
