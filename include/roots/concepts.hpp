/// @file roots/concepts.hpp
/// @brief Contracts for scalar root finding.
///
/// `num::scalar_function` and `num::differentiable_function` are defined in
/// `algebra/concepts.hpp`. What root finding adds is the bracket.
#pragma once

#include "algebra/concepts.hpp"
#include "core/types.hpp"
#include <concepts>

namespace num {

/// @brief Function whose sign can be compared across an interval.
///
/// \f[ f(a) \, f(b) < 0 \f]
///
/// A sign change guarantees a root by the intermediate value theorem, which is
/// what makes bisection and Brent's method unconditionally convergent. Whether a
/// given \f$[a,b]\f$ brackets a root depends on the values, so it is checked by
/// `num::roots::debug::verify_bracket`.
template <class F, class T = real>
concept bracketable_function = scalar_function<F, T> && std::totally_ordered<T>;

} // namespace num
