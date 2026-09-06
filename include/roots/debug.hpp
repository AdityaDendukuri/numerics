/// @file roots/debug.hpp
/// @brief Runtime verification of root-finding preconditions.
#pragma once

#include "core/debug.hpp"
#include "core/types.hpp"
#include "roots/concepts.hpp"
#include <cmath>
#include <source_location>
#include <string>

namespace num::roots::debug {

using num::debug::diagnostic_level;
using num::debug::get_level;
using num::debug::panic;

/// @brief Verify that \f$f\f$ changes sign on \f$[a,b]\f$.
///
/// Bisection and Brent's method converge because a sign change forces a root
/// inside the interval. Without one they return a value that is not a root and
/// report no error, so the bracket is checked rather than assumed.
template <class F, class T = real>
requires scalar_function<F, T> inline void
verify_bracket(F &&f, T a, T b, std::source_location loc = std::source_location::current()) {
    if (get_level() == diagnostic_level::off) {
        return;
    }
    if (!(a < b)) {
        panic("BracketError", "root bracket requires a < b", loc);
    }
    const T fa = f(a);
    const T fb = f(b);
    if (fa == T(0) || fb == T(0)) {
        return;
    }
    if ((fa > T(0)) == (fb > T(0))) {
        panic("BracketError",
              "f does not change sign on the bracket: f(a) = " +
                  std::to_string(static_cast<double>(fa)) +
                  ", f(b) = " + std::to_string(static_cast<double>(fb)) +
                  ". A bracketing method cannot converge to a root here.",
              loc);
    }
}

/// @brief Verify a supplied derivative against a central difference.
///
/// A wrong derivative makes Newton's method converge slowly, to the wrong point,
/// or not at all, and none of those report an error on their own.
template <class F, class D, class T = real>
requires differentiable_function<F, D, T> inline void
verify_derivative(F &&f, D &&df, T x, T tol = T(1e-5),
                  std::source_location loc = std::source_location::current()) {
    if constexpr (!num::debug::sampling_compiled_in) {
        return;
    }
    if (num::debug::get_level() != num::debug::diagnostic_level::full) {
        return;
    }
    const T h = std::cbrt(std::numeric_limits<T>::epsilon()) * std::max(T(1), std::abs(x));
    const T numeric = (f(x + h) - f(x - h)) / (T(2) * h);
    const T supplied = df(x);
    const T scale = std::max(std::abs(numeric), std::abs(supplied)) + std::numeric_limits<T>::min();
    if (std::abs(numeric - supplied) / scale > tol) {
        panic("DerivativeError",
              "supplied derivative disagrees with a central difference at x = " +
                  std::to_string(static_cast<double>(x)) + ": got " +
                  std::to_string(static_cast<double>(supplied)) + ", expected about " +
                  std::to_string(static_cast<double>(numeric)),
              loc);
    }
}

} // namespace num::roots::debug
