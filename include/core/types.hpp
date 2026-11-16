/// @file types.hpp
/// @brief The scalar, index and container aliases used across the library.
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

/// @brief The floating-point type of every real quantity, `double`.
using real = double;
using idx = std::size_t;
/// @brief The complex type over `num::real`.
using cplx = std::complex<real>;

/// @name Floating-point constants
///
/// `pi`, `e` and the rest come from `std::numbers`; `eps`, `inf` and `nan` from
/// `std::numeric_limits`. Each has a `_v<T>` form for precisions other than `real`. Unqualified,
/// `nan` is ambiguous with the C library function `::nan`, so write `num::nan`.
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
/// @brief The gap between 1 and the next representable value, \f$2^{-52}\f$ for `double`.
template <std::floating_point T>
inline constexpr T eps_v = std::numeric_limits<T>::epsilon();
/// @brief The unit roundoff \f$u = \epsilon/2\f$, the bound on the relative error of one rounding.
template <std::floating_point T>
inline constexpr T unit_roundoff_v = std::numeric_limits<T>::epsilon() / T(2);
template <std::floating_point T>
inline constexpr T inf_v = std::numeric_limits<T>::infinity();
template <std::floating_point T>
inline constexpr T nan_v = std::numeric_limits<T>::quiet_NaN();
/// @brief The smallest positive normal value; below it precision degrades gradually.
template <std::floating_point T>
inline constexpr T tiny_v = std::numeric_limits<T>::min();
template <std::floating_point T>
inline constexpr T huge_v = std::numeric_limits<T>::max();

inline constexpr real pi = pi_v<real>;
inline constexpr real two_pi = two_pi_v<real>;
inline constexpr real half_pi = half_pi_v<real>;
inline constexpr real inv_pi = inv_pi_v<real>;
inline constexpr real e = e_v<real>;
/// @brief The golden ratio.
inline constexpr real phi = phi_v<real>;
inline constexpr real sqrt2 = sqrt2_v<real>;
inline constexpr real sqrt3 = sqrt3_v<real>;
inline constexpr real ln2 = ln2_v<real>;
inline constexpr real eps = eps_v<real>;
inline constexpr real unit_roundoff = unit_roundoff_v<real>;
inline constexpr real inf = inf_v<real>;
inline constexpr real nan = nan_v<real>;
inline constexpr real tiny = tiny_v<real>;
inline constexpr real huge = huge_v<real>;

/// @}

/// @name Floating-point comparisons
///
/// `std::max(m, x)` returns `m` when `x` is NaN, because every comparison with NaN is false, so a
/// running maximum silently discards a NaN. `num::max` and `num::min` return NaN if either
/// argument is NaN, like IEEE 754-2019 `maximum` and `minimum`.
/// @{

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

/// @brief Whether `a` and `b` agree to a relative tolerance `rtol` or an absolute one `atol`:
/// \f$|a-b| \le \max(\text{rtol}\cdot\max(|a|,|b|),\ \text{atol})\f$.
///
/// Python's `math.isclose`. Equal infinities are close; NaN is close to nothing. The default
/// `atol = 0` makes a comparison against zero fail unless `a == b`: pass the scale at which a
/// value counts as zero.
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

/// @brief The number of representable values between `a` and `b`: 0 when equal, 1 when adjacent.
///
/// The distance to use for "correct to within k ulps"; `+0` and `-0` are 0 apart. The maximum
/// `uint64_t` if either argument is NaN.
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

/// @brief A stopping tolerance: an error passes when
/// \f$\text{error} \le \max(\text{rtol}\cdot|\text{scale}|,\ \text{atol})\f$.
///
/// Every iterative routine states what its error and scale are: for a linear solver
/// \f$\|b - Ax\|_2\f$ against \f$\|b\|_2\f$, for a root finder the last step against \f$|x|\f$,
/// for a quadrature rule the error estimate against the integral. `rtol` makes the test
/// invariant to rescaling the problem; `atol` is the floor for a scale that is zero or
/// meaningless, such as \f$b = 0\f$ or a root at the origin. NaN never passes.
///
/// Set the fields by name: `{.rtol = 1e-8}`, `{.rtol = 1e-8, .atol = 1e-14}`. A field left out
/// is zero, so `{.atol = 1e-12}` is a purely absolute test.
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
    /// @brief Whether the tolerance can be met at all: finite, nonnegative, not both zero.
    [[nodiscard]] constexpr bool valid() const noexcept {
        return rtol >= T(0) && atol >= T(0) && (rtol > T(0) || atol > T(0)) &&
               rtol < inf_v<T> && atol < inf_v<T>;
    }
};

using tolerance = basic_tolerance<real>;

/// @name Container vocabulary
///
/// Alias templates, not wrappers: `num::array<T>` is `std::vector<T>`. `array`, `static_array`
/// and `view` free the words vector and span for mathematics; the associative containers keep
/// their C++ names.
///
/// Fixed and dynamic extent get two names because an alias selecting between them through a
/// trait would make `template <class T> void f(array<T> &)` a non-deduced context.
/// @{

/// @brief A growable array. Storage, not an element of a vector space.
template <class T, class Alloc = std::allocator<T>>
using array = std::vector<T, Alloc>;

/// @brief An array whose length is fixed at compile time.
template <class T, std::size_t N>
using static_array = std::array<T, N>;

/// @brief A non-owning window onto contiguous storage.
template <class T, std::size_t Extent = std::dynamic_extent>
using view = std::span<T, Extent>;

/// @brief `std::unordered_map`: a hash map with no key order, like Rust `HashMap`.
template <class K, class V, class Hash = std::hash<K>, class Eq = std::equal_to<K>,
          class Alloc = std::allocator<std::pair<const K, V>>>
using unordered_map = std::unordered_map<K, V, Hash, Eq, Alloc>;

/// @brief `std::map`: a map sorted by key, like Rust `BTreeMap`.
template <class K, class V, class Compare = std::less<K>,
          class Alloc = std::allocator<std::pair<const K, V>>>
using map = std::map<K, V, Compare, Alloc>;

/// @brief `std::unordered_set`: a hash set with no key order, like Rust `HashSet`.
template <class K, class Hash = std::hash<K>, class Eq = std::equal_to<K>,
          class Alloc = std::allocator<K>>
using unordered_set = std::unordered_set<K, Hash, Eq, Alloc>;

/// @brief `std::set`: a set sorted by key, like Rust `BTreeSet`.
template <class K, class Compare = std::less<K>, class Alloc = std::allocator<K>>
using set = std::set<K, Compare, Alloc>;

/// @brief Add an element at the end of an array, constructed from `args`. It is `emplace_back`,
/// named for what the caller does, and returns a reference to the new element.
template <class T, class Alloc, class... Args>
T &append(array<T, Alloc> &values, Args &&...args) {
    return values.emplace_back(std::forward<Args>(args)...);
}

/// @}

/// @brief Cast any integer to idx without a verbose static_cast.
template <class T>
constexpr idx to_idx(T x) noexcept {
    return static_cast<idx>(x);
}

/// scalar callback \f$f(x)\f$.
using scalar_fn = std::function<real(real)>;

/// vec callback writing \f$f(t, y)\f$ into a caller-provided buffer.
using vector_fn = std::function<void(real, real *, real *)>;

} // namespace num
