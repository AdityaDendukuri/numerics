/// @file quadrature/concepts.hpp
/// @brief Contracts for quadrature rules.
///
/// The integrand itself is an `num::scalar_function`, defined in
/// `algebra/concepts.hpp` because a map on a scalar field is algebra vocabulary
/// rather than something quadrature introduces.
#pragma once

#include "algebra/concepts.hpp"
#include "core/types.hpp"
#include <concepts>

namespace num {

/// @brief Rule supplying nodes \f$s_k\f$ and weights \f$w_k\f$ on a complex contour.
///
/// Used for inverse Laplace transforms, where the integral runs along a contour
/// rather than an interval. The rule never sees the integrand: the caller
/// evaluates the transform at each node and accumulates.
template <class R, class T = real>
concept contour_rule = field<T> && requires(const R &rule, T t) {
    { rule.nodes(t) };
};

} // namespace num
