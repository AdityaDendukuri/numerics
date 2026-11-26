/// @file spatial/concepts.hpp
/// @brief Concepts for spatial structures and smoothing kernels.
#pragma once

#include "algebra/concepts.hpp"
#include "core/index_space.hpp"
#include "core/types.hpp"
#include <concepts>
#include <utility>

namespace num {

/// @brief Function `pos(i)` that returns the position \f$(x_i, y_i)\f$.
template <class A, class Scalar = real>
concept position_accessor_2d = field<Scalar> && requires(const A &get_pos, int i) {
    { get_pos(i) } -> std::convertible_to<std::pair<Scalar, Scalar>>;
};

/// @brief Structure whose `query(p, f)` calls `f(j)` for each particle that may be near \f$p\f$.
template <class L, class Scalar = real>
concept neighbor_query_2d = field<Scalar> &&
    requires(const L &list, Scalar px, Scalar py) {
    list.query(px, py, [](int) {});
};

/// @brief Smoothing kernel: \f$W(r, h)\f$ and \f$\partial W / \partial r\f$.
template <class K, class Scalar = float>
concept smoothing_kernel = field<Scalar> && requires(Scalar r, Scalar h) {
    { K::W(r, h) } -> std::convertible_to<Scalar>;
    { K::dW_dr(r, h) } -> std::convertible_to<Scalar>;
};


} // namespace num
