/// @file math.hpp
/// @brief Thin wrappers around C++17 `<cmath>` and `<numeric>` with readable names.
///
/// The standard library names for special functions (cyl_bessel_j, comp_ellint_1,
/// cyl_neumann, etc.) are accurate but opaque. This header re-exports them under
/// the names used in textbooks and mathematical literature.
///
/// Random number generation wraps the mt19937 boilerplate into a plain struct.
///
/// Include this header instead of pulling `<cmath>` special functions directly.
#pragma once

#include "core/types.hpp"
#include <cmath>
#include <numeric>
#include <random>
#include <vector>
#include <cassert>

// Apple libc++ does not implement C++17 special math functions.
// Use Boost.Math as a drop-in replacement when building on macOS.
#ifdef NUMERICS_USE_BOOST_MATH
#  include <boost/math/special_functions/bessel.hpp>
#  include <boost/math/special_functions/legendre.hpp>
#  include <boost/math/special_functions/hermite.hpp>
#  include <boost/math/special_functions/laguerre.hpp>
#  include <boost/math/special_functions/ellint_1.hpp>
#  include <boost/math/special_functions/ellint_2.hpp>
#  include <boost/math/special_functions/ellint_3.hpp>
#  include <boost/math/special_functions/expint.hpp>
#  include <boost/math/special_functions/zeta.hpp>
#  include <boost/math/special_functions/beta.hpp>
#  include <boost/math/special_functions/spherical_harmonic.hpp>
#endif

namespace num {

// Mathematical constants  (C++20 <numbers> not available in C++17)

constexpr real pi      = 3.14159265358979323846;
constexpr real e       = 2.71828182845904523536;
constexpr real phi     = 1.61803398874989484820;  ///< Golden ratio
constexpr real sqrt2   = 1.41421356237309504880;
constexpr real sqrt3   = 1.73205080756887729353;
constexpr real ln2     = 0.69314718055994530942;
constexpr real inv_pi  = 0.31830988618379067154;  ///< 1/pi
constexpr real two_pi  = 6.28318530717958647692;  ///< 2pi
constexpr real half_pi = 1.57079632679489661923;  ///< pi/2

// Cylindrical Bessel functions  (C++17, <cmath>)

/// @brief J_nu(x)  -- Bessel function of the first kind
inline real bessel_j(real nu, real x) {
#ifdef NUMERICS_USE_BOOST_MATH
    return boost::math::cyl_bessel_j(nu, x);
#else
    return std::cyl_bessel_j(nu, x);
#endif
}

/// @brief Y_nu(x)  -- Bessel function of the second kind (Neumann function)
inline real bessel_y(real nu, real x) {
#ifdef NUMERICS_USE_BOOST_MATH
    return boost::math::cyl_neumann(nu, x);
#else
    return std::cyl_neumann(nu, x);
#endif
}

/// @brief I_nu(x)  -- modified Bessel function of the first kind
inline real bessel_i(real nu, real x) {
#ifdef NUMERICS_USE_BOOST_MATH
    return boost::math::cyl_bessel_i(nu, x);
#else
    return std::cyl_bessel_i(nu, x);
#endif
}

/// @brief K_nu(x)  -- modified Bessel function of the second kind
inline real bessel_k(real nu, real x) {
#ifdef NUMERICS_USE_BOOST_MATH
    return boost::math::cyl_bessel_k(nu, x);
#else
    return std::cyl_bessel_k(nu, x);
#endif
}

// Spherical Bessel functions  (C++17, <cmath>)

/// @brief j_n(x)  -- spherical Bessel function of the first kind
inline real sph_bessel_j(unsigned int n, real x) {
#ifdef NUMERICS_USE_BOOST_MATH
    return boost::math::sph_bessel(n, x);
#else
    return std::sph_bessel(n, x);
#endif
}

/// @brief y_n(x)  -- spherical Neumann function (spherical Bessel of the second kind)
inline real sph_bessel_y(unsigned int n, real x) {
#ifdef NUMERICS_USE_BOOST_MATH
    return boost::math::sph_neumann(n, x);
#else
    return std::sph_neumann(n, x);
#endif
}

// Orthogonal polynomials  (C++17, <cmath>)

/// @brief P_n(x)  -- Legendre polynomial of degree n
inline real legendre(unsigned int n, real x) {
#ifdef NUMERICS_USE_BOOST_MATH
    return boost::math::legendre_p(static_cast<int>(n), x);
#else
    return std::legendre(n, x);
#endif
}

/// @brief P_n^m(x)  -- associated Legendre polynomial
inline real assoc_legendre(unsigned int n, unsigned int m, real x) {
#ifdef NUMERICS_USE_BOOST_MATH
    return boost::math::legendre_p(static_cast<int>(n), static_cast<int>(m), x);
#else
    return std::assoc_legendre(n, m, x);
#endif
}

/// @brief Y_l^m(theta)  -- spherical harmonic (real part, theta in radians)
inline real sph_legendre(unsigned int l, unsigned int m, real theta) {
#ifdef NUMERICS_USE_BOOST_MATH
    return boost::math::spherical_harmonic_r(l, static_cast<int>(m), theta, real(0));
#else
    return std::sph_legendre(l, m, theta);
#endif
}

/// @brief H_n(x)  -- (physicists') Hermite polynomial
inline real hermite(unsigned int n, real x) {
#ifdef NUMERICS_USE_BOOST_MATH
    return boost::math::hermite(n, x);
#else
    return std::hermite(n, x);
#endif
}

/// @brief L_n(x)  -- Laguerre polynomial
inline real laguerre(unsigned int n, real x) {
#ifdef NUMERICS_USE_BOOST_MATH
    return boost::math::laguerre(n, x);
#else
    return std::laguerre(n, x);
#endif
}

/// @brief L_n^m(x)  -- associated Laguerre polynomial
inline real assoc_laguerre(unsigned int n, unsigned int m, real x) {
#ifdef NUMERICS_USE_BOOST_MATH
    return boost::math::laguerre(n, m, x);
#else
    return std::assoc_laguerre(n, m, x);
#endif
}

// Elliptic integrals  (C++17, <cmath>)
// Names: K(k), E(k), Pi(n,k) for complete; F(k,phi), E(k,phi), Pi(n,k,phi) for incomplete

/// @brief K(k)  -- complete elliptic integral of the first kind
inline real ellint_K(real k) {
#ifdef NUMERICS_USE_BOOST_MATH
    return boost::math::ellint_1(k);
#else
    return std::comp_ellint_1(k);
#endif
}

/// @brief E(k)  -- complete elliptic integral of the second kind
inline real ellint_E(real k) {
#ifdef NUMERICS_USE_BOOST_MATH
    return boost::math::ellint_2(k);
#else
    return std::comp_ellint_2(k);
#endif
}

/// @brief Pi(n, k)  -- complete elliptic integral of the third kind
inline real ellint_Pi(real n, real k) {
#ifdef NUMERICS_USE_BOOST_MATH
    return boost::math::ellint_3(k, n);   // boost: (k, n); std: (n, k)
#else
    return std::comp_ellint_3(n, k);
#endif
}

/// @brief F(k, phi)  -- incomplete elliptic integral of the first kind
inline real ellint_F(real k, real phi) {
#ifdef NUMERICS_USE_BOOST_MATH
    return boost::math::ellint_1(k, phi);
#else
    return std::ellint_1(k, phi);
#endif
}

/// @brief E(k, phi)  -- incomplete elliptic integral of the second kind
inline real ellint_Ei(real k, real phi) {
#ifdef NUMERICS_USE_BOOST_MATH
    return boost::math::ellint_2(k, phi);
#else
    return std::ellint_2(k, phi);
#endif
}

/// @brief Pi(n, k, phi)  -- incomplete elliptic integral of the third kind
inline real ellint_Pi_inc(real n, real k, real phi) {
#ifdef NUMERICS_USE_BOOST_MATH
    return boost::math::ellint_3(k, n, phi);  // boost: (k, n, phi); std: (n, k, phi)
#else
    return std::ellint_3(n, k, phi);
#endif
}

// Other special functions  (C++17, <cmath>)

/// @brief Ei(x)  -- exponential integral
inline real expint(real x) {
#ifdef NUMERICS_USE_BOOST_MATH
    return boost::math::expint(x);
#else
    return std::expint(x);
#endif
}

/// @brief zeta(x)  -- Riemann zeta function
inline real zeta(real x) {
#ifdef NUMERICS_USE_BOOST_MATH
    return boost::math::zeta(x);
#else
    return std::riemann_zeta(x);
#endif
}

/// @brief B(a, b)  -- beta function
inline real beta(real a, real b) {
#ifdef NUMERICS_USE_BOOST_MATH
    return boost::math::beta(a, b);
#else
    return std::beta(a, b);
#endif
}

// Sequence utilities  (wrapping <numeric>)

/// @brief Evenly spaced values from start to stop, inclusive. MATLAB/NumPy linspace.
inline std::vector<real> linspace(real start, real stop, idx n) {
    assert(n >= 2);
    std::vector<real> out(n);
    real step = (stop - start) / static_cast<real>(n - 1);
    for (idx i = 0; i < n; ++i)
        out[i] = start + static_cast<real>(i) * step;
    return out;
}

/// @brief Integer sequence [start, start+1, ..., start+n-1]. Wraps std::iota.
inline std::vector<int> int_range(int start, int n) {
    assert(n >= 0);
    std::vector<int> out(static_cast<idx>(n));
    std::iota(out.begin(), out.end(), start);
    return out;
}

// Random number generation  (wrapping the mt19937 boilerplate)

/// @brief Seeded pseudo-random number generator (Mersenne Twister).
///        Pass a pointer to rng_* functions to draw samples.
struct Rng {
    std::mt19937 engine;

    explicit Rng(uint32_t seed) : engine(seed) {}

    /// Seed from hardware entropy.
    Rng() : engine(std::random_device{}()) {}
};

/// @brief Uniform real in [lo, hi).
inline real rng_uniform(Rng* r, real lo, real hi) {
    return std::uniform_real_distribution<real>{lo, hi}(r->engine);
}

/// @brief Normal (Gaussian) sample with given mean and standard deviation.
inline real rng_normal(Rng* r, real mean, real stddev) {
    return std::normal_distribution<real>{mean, stddev}(r->engine);
}

/// @brief Uniform integer in [lo, hi] (inclusive on both ends).
inline int rng_int(Rng* r, int lo, int hi) {
    return std::uniform_int_distribution<int>{lo, hi}(r->engine);
}

} // namespace num
