/// @file math.hpp
/// @brief Thin wrappers around `<cmath>` and standard mathematical functions.
///
/// Standard orthogonal polynomials, Bessel functions, and constants with
/// zero external dependencies.
#pragma once

#include "core/types.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

namespace num {

// Mathematical constants
constexpr real pi = 3.14159265358979323846;
constexpr real e = 2.71828182845904523536;
constexpr real phi = 1.61803398874989484820; ///< Golden ratio
constexpr real sqrt2 = 1.41421356237309504880;
constexpr real sqrt3 = 1.73205080756887729353;
constexpr real ln2 = 0.69314718055994530942;
constexpr real inv_pi = 0.31830988618379067154;  ///< 1/pi
constexpr real two_pi = 6.28318530717958647692;  ///< 2pi
constexpr real half_pi = 1.57079632679489661923; ///< pi/2

// Cylindrical Bessel functions (POSIX / C99 / exact series)

/// @brief J_nu(x) -- Bessel function of the first kind
inline real bessel_j(real nu, real x) {
    int n = static_cast<int>(std::round(nu));
    if (std::abs(nu - static_cast<real>(n)) < 1e-12) {
        if (n == 0) {
            return ::j0(x);
        }
        if (n == 1) {
            return ::j1(x);
        }
        return ::jn(n, x);
    }
    // Small argument Taylor series fallback for general nu
    real sum = 0.0;
    real term = std::pow(0.5 * x, nu) / std::tgamma(nu + 1.0);
    real x2_4 = -0.25 * x * x;
    for (int k = 0; k < 30; ++k) {
        sum += term;
        term *= x2_4 / ((k + 1.0) * (nu + k + 1.0));
        if (std::abs(term) < 1e-16 * std::abs(sum)) {
            break;
        }
    }
    return sum;
}

/// @brief Y_nu(x) -- Bessel function of the second kind (Neumann function)
inline real bessel_y(real nu, real x) {
    int n = static_cast<int>(std::round(nu));
    if (n == 0) {
        return ::y0(x);
    }
    if (n == 1) {
        return ::y1(x);
    }
    return ::yn(n, x);
}

/// @brief I_nu(x) -- modified Bessel function of the first kind
inline real bessel_i(real nu, real x) {
    real sum = 0.0;
    real term = std::pow(0.5 * x, nu) / std::tgamma(nu + 1.0);
    real x2_4 = 0.25 * x * x;
    for (int k = 0; k < 40; ++k) {
        sum += term;
        term *= x2_4 / ((k + 1.0) * (nu + k + 1.0));
        if (std::abs(term) < 1e-16 * std::abs(sum)) {
            break;
        }
    }
    return sum;
}

// Spherical Bessel functions

/// @brief j_n(x) -- spherical Bessel function of the first kind
inline real sph_bessel_j(unsigned int n, real x) {
    if (std::abs(x) < 1e-14) {
        return (n == 0) ? 1.0 : 0.0;
    }
    if (n == 0) {
        return std::sin(x) / x;
    }
    if (n == 1) {
        return (std::sin(x) / (x * x)) - (std::cos(x) / x);
    }
    real j_prev = std::sin(x) / x;
    real j_curr = (std::sin(x) / (x * x)) - (std::cos(x) / x);
    for (unsigned int k = 1; k < n; ++k) {
        real j_next = (((2.0 * k) + 1.0) / x) * j_curr - j_prev;
        j_prev = j_curr;
        j_curr = j_next;
    }
    return j_curr;
}

/// @brief y_n(x) -- spherical Neumann function
inline real sph_bessel_y(unsigned int n, real x) {
    if (n == 0) {
        return -std::cos(x) / x;
    }
    if (n == 1) {
        return (-std::cos(x) / (x * x)) - (std::sin(x) / x);
    }
    real y_prev = -std::cos(x) / x;
    real y_curr = (-std::cos(x) / (x * x)) - (std::sin(x) / x);
    for (unsigned int k = 1; k < n; ++k) {
        real y_next = (((2.0 * k) + 1.0) / x) * y_curr - y_prev;
        y_prev = y_curr;
        y_curr = y_next;
    }
    return y_curr;
}

// Orthogonal polynomials (evaluated via stable 3-term recurrence relations)

/// @brief P_n(x) -- Legendre polynomial of degree n
inline real legendre(unsigned int n, real x) {
    if (n == 0) {
        return 1.0;
    }
    if (n == 1) {
        return x;
    }
    real p_prev = 1.0;
    real p_curr = x;
    for (unsigned int k = 1; k < n; ++k) {
        real p_next = ((((2.0 * k) + 1.0) * x * p_curr) - (static_cast<real>(k) * p_prev)) /
                      static_cast<real>(k + 1);
        p_prev = p_curr;
        p_curr = p_next;
    }
    return p_curr;
}

/// @brief P_n^m(x) -- associated Legendre polynomial
inline real assoc_legendre(unsigned int n, unsigned int m, real x) {
    if (m > n || std::abs(x) > 1.0) {
        return 0.0;
    }
    // Compute P_m^m(x)
    real pmm = 1.0;
    if (m > 0) {
        real somx2 = std::sqrt((1.0 - x) * (1.0 + x));
        real fact = 1.0;
        for (unsigned int i = 1; i <= m; ++i) {
            pmm *= -fact * somx2;
            fact += 2.0;
        }
    }
    if (n == m) {
        return pmm;
    }
    // Compute P_{m+1}^m(x)
    real pmp1 = x * ((2.0 * m) + 1.0) * pmm;
    if (n == m + 1) {
        return pmp1;
    }
    // Recurrence to P_n^m(x)
    real pll = 0.0;
    for (unsigned int ll = m + 2; ll <= n; ++ll) {
        pll = ((((2.0 * ll) - 1.0) * x * pmp1) - ((static_cast<real>(ll + m) - 1.0) * pmm)) /
              static_cast<real>(ll - m);
        pmm = pmp1;
        pmp1 = pll;
    }
    return pll;
}

/// @brief H_n(x) -- (physicists') Hermite polynomial
inline real hermite(unsigned int n, real x) {
    if (n == 0) {
        return 1.0;
    }
    if (n == 1) {
        return 2.0 * x;
    }
    real h_prev = 1.0;
    real h_curr = 2.0 * x;
    for (unsigned int k = 1; k < n; ++k) {
        real h_next = (2.0 * x * h_curr) - (2.0 * static_cast<real>(k) * h_prev);
        h_prev = h_curr;
        h_curr = h_next;
    }
    return h_curr;
}

/// @brief L_n(x) -- Laguerre polynomial
inline real laguerre(unsigned int n, real x) {
    if (n == 0) {
        return 1.0;
    }
    if (n == 1) {
        return 1.0 - x;
    }
    real l_prev = 1.0;
    real l_curr = 1.0 - x;
    for (unsigned int k = 1; k < n; ++k) {
        real l_next = ((((2.0 * k) + 1.0 - x) * l_curr) - (static_cast<real>(k) * l_prev)) /
                      static_cast<real>(k + 1);
        l_prev = l_curr;
        l_curr = l_next;
    }
    return l_curr;
}

/// @brief L_n^m(x) -- associated Laguerre polynomial
inline real assoc_laguerre(unsigned int n, unsigned int m, real x) {
    if (n == 0) {
        return 1.0;
    }
    if (n == 1) {
        return 1.0 + static_cast<real>(m) - x;
    }
    real l_prev = 1.0;
    real l_curr = 1.0 + static_cast<real>(m) - x;
    for (unsigned int k = 1; k < n; ++k) {
        real l_next =
            ((((2.0 * k) + 1.0 + static_cast<real>(m) - x) * l_curr) -
             ((static_cast<real>(k) + static_cast<real>(m)) * l_prev)) /
            static_cast<real>(k + 1);
        l_prev = l_curr;
        l_curr = l_next;
    }
    return l_curr;
}

/// @brief B(a, b) -- beta function (numerically stable via log-gamma)
inline real beta(real a, real b) {
    return std::exp(std::lgamma(a) + std::lgamma(b) - std::lgamma(a + b));
}

// Sequence utilities (wrapping <numeric>)

/// @brief Evenly spaced values from start to stop, inclusive. MATLAB/NumPy linspace.
inline std::vector<real> linspace(real start, real stop, idx n) {
    assert(n >= 2);
    std::vector<real> out(n);
    real step = (stop - start) / static_cast<real>(n - 1);
    for (idx i = 0; i < n; ++i) {
        out[i] = start + (static_cast<real>(i) * step);
    }
    return out;
}

/// @brief Values with evenly spaced exponents, inclusive.
inline std::vector<real> logspace(real start, real stop, idx n, real base = 10.0) {
    auto out = linspace(start, stop, n);
    for (real &exponent : out) {
        exponent = std::pow(base, exponent);
    }
    return out;
}

/// @brief Integer sequence [start, start+1, ..., start+n-1]. Wraps std::iota.
inline std::vector<int> int_range(int start, int n) {
    assert(n >= 0);
    std::vector<int> out(static_cast<idx>(n));
    std::iota(out.begin(), out.end(), start);
    return out;
}

// Random number generation (wrapping the mt19937 boilerplate)

/// @brief Seeded pseudo-random number generator (Mersenne Twister).
struct Rng {
    std::mt19937 engine;

    explicit Rng(uint32_t seed) : engine(seed) {}

    /// Seed from hardware entropy.
    Rng() : engine(std::random_device{}()) {}
};

/// @brief Uniform real in [lo, hi).
inline real rng_uniform(Rng *r, real lo, real hi) {
    return std::uniform_real_distribution<real>{lo, hi}(r->engine);
}

/// @brief Normal (Gaussian) sample with given mean and standard deviation.
inline real rng_normal(Rng *r, real mean, real stddev) {
    return std::normal_distribution<real>{mean, stddev}(r->engine);
}

/// @brief Uniform integer in [lo, hi] (inclusive on both ends).
inline int rng_int(Rng *r, int lo, int hi) {
    return std::uniform_int_distribution<int>{lo, hi}(r->engine);
}

// Spatial distributions

/// @brief 2D isotropic Gaussian centred at (cx, cy) with width sigma
inline real gaussian2d(real x, real y, real cx, real cy, real sigma) {
    real dx = x - cx, dy = y - cy;
    return std::exp(-((dx * dx) + (dy * dy)) / (2.0 * sigma * sigma));
}

} // namespace num
