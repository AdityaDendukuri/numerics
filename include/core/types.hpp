/// @file core/types.hpp
/// @brief Scalar, index and container aliases.
#pragma once

#include <array>
#include <bit>
#include <cmath>
#include <complex>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <numbers>
#include <set>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace num {

/// @brief Real scalar type: `double`.
using real = double;
using idx = std::size_t;
/// @brief Complex scalar type: `std::complex<real>`.
using cplx = std::complex<real>;

/// @name Floating-point constants
/// Each has a `_v<T>` form for other precisions.
/// @{

template <std::floating_point T>
inline constexpr T pi_v = std::numbers::pi_v<T>;
template <std::floating_point T>
inline constexpr T two_pi_v = T(2) * std::numbers::pi_v<T>;
template <std::floating_point T>
inline constexpr T e_v = std::numbers::e_v<T>;
template <std::floating_point T>
inline constexpr T half_pi_v = std::numbers::pi_v<T> / T(2);
template <std::floating_point T>
inline constexpr T inv_pi_v = std::numbers::inv_pi_v<T>;
template <std::floating_point T>
inline constexpr T phi_v = std::numbers::phi_v<T>;
template <std::floating_point T>
inline constexpr T sqrt2_v = std::numbers::sqrt2_v<T>;
template <std::floating_point T>
inline constexpr T sqrt3_v = std::numbers::sqrt3_v<T>;
template <std::floating_point T>
inline constexpr T ln2_v = std::numbers::ln2_v<T>;
/// @brief Machine epsilon: \f$2^{-52}\f$ for `double`.
template <std::floating_point T>
inline constexpr T eps_v = std::numeric_limits<T>::epsilon();
/// @brief Unit roundoff: \f$u = \epsilon / 2\f$.
template <std::floating_point T>
inline constexpr T unit_roundoff_v = std::numeric_limits<T>::epsilon() / T(2);
template <std::floating_point T>
inline constexpr T inf_v = std::numeric_limits<T>::infinity();
template <std::floating_point T>
inline constexpr T nan_v = std::numeric_limits<T>::quiet_NaN();
/// @brief Smallest positive normal value.
template <std::floating_point T>
inline constexpr T tiny_v = std::numeric_limits<T>::min();
template <std::floating_point T>
inline constexpr T huge_v = std::numeric_limits<T>::max();

/// @brief \f$\pi\f$.
inline constexpr real pi = pi_v<real>;
/// @brief \f$2\pi\f$.
inline constexpr real two_pi = two_pi_v<real>;
/// @brief \f$\pi/2\f$.
inline constexpr real half_pi = half_pi_v<real>;
/// @brief \f$1/\pi\f$.
inline constexpr real inv_pi = inv_pi_v<real>;
/// @brief Euler's number: \f$e\f$.
inline constexpr real e = e_v<real>;
/// @brief Golden ratio: \f$\varphi = (1 + \sqrt{5}) / 2\f$.
inline constexpr real phi = phi_v<real>;
/// @brief \f$\sqrt{2}\f$.
inline constexpr real sqrt2 = sqrt2_v<real>;
/// @brief \f$\sqrt{3}\f$.
inline constexpr real sqrt3 = sqrt3_v<real>;
/// @brief \f$\ln 2\f$.
inline constexpr real ln2 = ln2_v<real>;
/// @brief Machine epsilon: \f$2^{-52}\f$.
inline constexpr real eps = eps_v<real>;
/// @brief Unit roundoff: \f$2^{-53}\f$.
inline constexpr real unit_roundoff = unit_roundoff_v<real>;
/// @brief Positive infinity.
inline constexpr real inf = inf_v<real>;
/// @brief Quiet NaN. Qualify as `num::nan`; unqualified it clashes with `::nan`.
inline constexpr real nan = nan_v<real>;
/// @brief Smallest positive normal `real`.
inline constexpr real tiny = tiny_v<real>;
/// @brief The largest finite `real`.
inline constexpr real huge = huge_v<real>;

/// @}

/// @name Floating-point comparisons
/// NaN-propagating, unlike `std::max` and `std::min`.
/// @{

/// @brief Return the larger of `a` and `b`, or NaN if either is NaN.
template <class T>
[[nodiscard]] constexpr T max(T a, T b) noexcept {
    if constexpr (std::floating_point<T>) {
        if (a != a) {
            return a;
        }
        if (b != b) {
            return b;
        }
    }
    return a < b ? b : a;
}

/// @brief Return the smaller of `a` and `b`, or NaN if either is NaN.
template <class T>
[[nodiscard]] constexpr T min(T a, T b) noexcept {
    if constexpr (std::floating_point<T>) {
        if (a != a) {
            return a;
        }
        if (b != b) {
            return b;
        }
    }
    return b < a ? b : a;
}

/// @brief Test whether two numbers are close:
/// \f$|a - b| \le \max(\mathrm{rtol} \max(|a|, |b|),\ \mathrm{atol})\f$.
/// With the default `atol = 0`, only zero is close to zero.
template <std::floating_point T>
[[nodiscard]] inline bool close(T a, T b, T rtol = T(1e-9), T atol = T(0)) noexcept {
    if (a == b) {
        return true;
    }
    if (!std::isfinite(a) || !std::isfinite(b)) {
        return false;
    }
    const T diff = std::abs(a - b);
    return diff <= rtol * std::max(std::abs(a), std::abs(b)) || diff <= atol;
}

/// @brief Count the representable values between `a` and `b`; the largest `uint64_t` for NaN.
/// @note `+0` and `-0` are 0 apart.
[[nodiscard]] inline std::uint64_t ulp_distance(double a, double b) noexcept {
    if (a != a || b != b) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    // Map the sign-magnitude bit pattern onto a monotone unsigned line.
    auto key = [](double x) {
        const auto bits = std::bit_cast<std::uint64_t>(x);
        constexpr std::uint64_t sign = std::uint64_t{1} << 63;
        return (bits & sign) != 0 ? sign - (bits & ~sign) : sign + bits;
    };
    const std::uint64_t ka = key(a), kb = key(b);
    return ka > kb ? ka - kb : kb - ka;
}

/// @}

/// @brief Stopping tolerance: \f$e \le \max(\mathrm{rtol}\,|s|,\ \mathrm{atol})\f$.
/// \f$e\f$ is the error and \f$s\f$ its scale.
/// Set fields by name, e.g. `{.rtol = 1e-8}`; omitted fields are zero.
/// @note NaN never passes.
template <std::floating_point T>
struct basic_tolerance {
    T rtol = T(0);
    T atol = T(0);

    [[nodiscard]] constexpr T bound(T scale) const noexcept {
        return num::max(rtol * (scale < T(0) ? -scale : scale), atol);
    }
    [[nodiscard]] constexpr bool met(T error, T scale) const noexcept {
        return error <= bound(scale);
    }
    /// @brief Test whether the tolerance can be met: finite, nonnegative, not both zero.
    [[nodiscard]] constexpr bool valid() const noexcept {
        return rtol >= T(0) && atol >= T(0) && (rtol > T(0) || atol > T(0)) &&
               rtol < inf_v<T> && atol < inf_v<T>;
    }
};

using tolerance = basic_tolerance<real>;

/// @name Container vocabulary
/// Aliases of the standard containers.
/// @{

/// @brief Growable array: `std::vector`.
template <class T, class Alloc = std::allocator<T>>
using array = std::vector<T, Alloc>;

/// @brief Fixed-size array: `std::array`.
template <class T, std::size_t N>
using static_array = std::array<T, N>;

/// @brief Non-owning contiguous view: `std::span`.
template <class T, std::size_t Extent = std::dynamic_extent>
using view = std::span<T, Extent>;

/// @brief Hash map: `std::unordered_map`.
template <class K, class V, class Hash = std::hash<K>, class Eq = std::equal_to<K>,
          class Alloc = std::allocator<std::pair<const K, V>>>
using unordered_map = std::unordered_map<K, V, Hash, Eq, Alloc>;

/// @brief Ordered map: `std::map`.
template <class K, class V, class Compare = std::less<K>,
          class Alloc = std::allocator<std::pair<const K, V>>>
using map = std::map<K, V, Compare, Alloc>;

/// @brief Hash set: `std::unordered_set`.
template <class K, class Hash = std::hash<K>, class Eq = std::equal_to<K>,
          class Alloc = std::allocator<K>>
using unordered_set = std::unordered_set<K, Hash, Eq, Alloc>;

/// @brief Ordered set: `std::set`.
template <class K, class Compare = std::less<K>, class Alloc = std::allocator<K>>
using set = std::set<K, Compare, Alloc>;

/// @brief Append an element constructed from `args`; returns a reference to it.
template <class T, class Alloc, class... Args>
T &append(array<T, Alloc> &values, Args &&...args) {
    return values.emplace_back(std::forward<Args>(args)...);
}

/// @}

/// @brief Convert an integer to `idx`.
template <class T>
constexpr idx to_idx(T x) noexcept {
    return static_cast<idx>(x);
}

/// @brief Scalar function: \f$f : \mathbb{R} \to \mathbb{R}\f$.
using scalar_fn = std::function<real(real)>;

/// @brief Vector function: writes \f$f(t, y)\f$ into a caller buffer.
using vector_fn = std::function<void(real, real *, real *)>;

} // namespace num
