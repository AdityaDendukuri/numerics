/// @file quadrature/concepts.hpp
/// @brief Contracts for quadrature rules.
///
/// The integrand itself is an `num::ScalarFunction`, defined in
/// `algebra/concepts.hpp` because a map on a scalar field is algebra vocabulary
/// rather than something quadrature introduces.
#pragma once

#include "algebra/concepts.hpp"
#include "core/types.hpp"
#include <concepts>

namespace num {

/// @brief Rule approximating \f$\int_a^b f\f$ on a finite interval.
///
/// A rule reports one value for a bounded interval. Adaptive rules satisfy this
/// too, since the refinement is internal to the call.
template <class R, class F, class T = real>
concept QuadratureRule = ScalarFunction<F, T> && requires(const R &rule, F f, T a, T b) {
    { rule(f, a, b) } -> std::convertible_to<T>;
};

/// @brief Rule supplying nodes \f$s_k\f$ and weights \f$w_k\f$ on a complex contour.
///
/// Used for inverse Laplace transforms, where the integral runs along a contour
/// rather than an interval. The rule never sees the integrand: the caller
/// evaluates the transform at each node and accumulates.
template <class R>
concept ContourRule = requires(const R &rule, real t) {
    { rule.nodes(t) };
};

} // namespace num
