/// @file algebra/scalar.hpp
/// @brief Scalar helpers shared by algorithms and runtime invariant sampling.
///
/// Numerical structure is stated over a scalar field \f$\mathbb{K}\f$ (typically
/// \f$\mathbb{R}\f$ or \f$\mathbb{C}\f$). These traits let structure and diagnostics
/// be written once and hold for both, rather than being fixed to `double`.
#pragma once

#include "core/math/associated.hpp"
#include "core/math/concepts.hpp"
#include <cmath>
#include <complex>
#include <concepts>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>

namespace num::scalars {

/// @brief Complex conjugation; the identity on real fields.
template <class T>
[[nodiscard]] constexpr T conj(const T &x) noexcept {
    if constexpr (is_complex_v<T>) {
        return std::conj(x);
    } else {
        return x;
    }
}

/// @brief Real part; the identity on real fields.
template <class T>
[[nodiscard]] constexpr real_t<T> re(const T &x) noexcept {
    if constexpr (is_complex_v<T>) {
        return x.real();
    } else {
        return x;
    }
}

/// @brief Modulus \f$|x|\f$, uniform over real and complex fields.
template <class T>
[[nodiscard]] inline real_t<T> mag(const T &x) noexcept {
    if constexpr (is_complex_v<T>) {
        return std::abs(x);
    } else {
        return x < T(0) ? -x : x;
    }
}

/// @brief Machine epsilon of the underlying real field.
template <class T>
[[nodiscard]] constexpr real_t<T> eps() noexcept {
    return std::numeric_limits<real_t<T>>::epsilon();
}

/// @brief Default relative tolerance for sampled property tests.
///
/// \f$\sqrt{\varepsilon}\f$ is the standard choice for randomized numerical probes:
/// tight enough to reject genuine violations, loose enough to absorb the
/// \f$O(n\varepsilon)\f$ round-off of the operator applications used to probe.
template <class T>
[[nodiscard]] inline real_t<T> sampling_tol() noexcept {
    return std::sqrt(eps<T>());
}

} // namespace num::scalars

namespace num {

namespace detail {

template <class A, class = void>
struct entry_of {
    using type = void;
};

template <class A>
struct entry_of<A, std::void_t<decltype(std::declval<const A &>()(std::size_t{0}, std::size_t{0}))>> {
    using type = std::remove_cvref_t<decltype(std::declval<const A &>()(std::size_t{0},
                                                                       std::size_t{0}))>;
};

} // namespace detail

using math::scalar_t;

/// @brief Entry type of a two-index container, or void when it has no such access.
template <class A>
using entry_t = typename detail::entry_of<A>::type;

using scalars::real_t;

} // namespace num
