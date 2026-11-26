/// @file container/util/math.hpp
/// @brief Special functions, orthogonal polynomials, ranges and random sampling.
#pragma once

#include "container/vector.hpp"
#include "core/types.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

namespace num {

// Cylindrical Bessel functions (POSIX / C99 / exact series)

/// @brief Evaluate the Bessel function of the first kind: \f$J_\nu(x)\f$.
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
        if (std::abs(term) < num::eps * std::abs(sum)) {
            break;
        }
    }
    return sum;
}

/// @brief Evaluate the Bessel function of the second kind: \f$Y_\nu(x)\f$.
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

/// @brief Evaluate the modified Bessel function of the first kind: \f$I_\nu(x)\f$.
inline real bessel_i(real nu, real x) {
    real sum = 0.0;
    real term = std::pow(0.5 * x, nu) / std::tgamma(nu + 1.0);
    real x2_4 = 0.25 * x * x;
    for (int k = 0; k < 40; ++k) {
        sum += term;
        term *= x2_4 / ((k + 1.0) * (nu + k + 1.0));
        if (std::abs(term) < num::eps * std::abs(sum)) {
            break;
        }
    }
    return sum;
}

// Spherical Bessel functions

/// @brief Evaluate the spherical Bessel function of the first kind: \f$j_n(x)\f$.
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

/// @brief Evaluate the spherical Bessel function of the second kind: \f$y_n(x)\f$.
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

/// @brief Evaluate the Legendre polynomial: \f$P_n(x)\f$.
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

/// @brief Evaluate the associated Legendre polynomial: \f$P_n^m(x)\f$.
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

/// @brief Evaluate the physicists' Hermite polynomial: \f$H_n(x)\f$.
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

/// @brief Evaluate the Laguerre polynomial: \f$L_n(x)\f$.
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

/// @brief Evaluate the associated Laguerre polynomial: \f$L_n^m(x)\f$.
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

/// @brief Evaluate the beta function: \f$B(a, b) = \Gamma(a) \Gamma(b) / \Gamma(a + b)\f$.
inline real beta(real a, real b) {
    return std::exp(std::lgamma(a) + std::lgamma(b) - std::lgamma(a + b));
}

// Sequence utilities (wrapping <numeric>)

/// @brief Return `n` evenly spaced values from `start` to `stop`, inclusive.
[[nodiscard]] inline vec<real> linspace(real start, real stop, idx n) {
    assert(n >= 2);
    vec<real> out(n);
    real step = (stop - start) / static_cast<real>(n - 1);
    for (idx i = 0; i < n; ++i) {
        out[i] = start + (static_cast<real>(i) * step);
    }
    return out;
}

/// @brief Return `n` values \f$b^t\f$ with \f$t\f$ evenly spaced in `[start, stop]`.
[[nodiscard]] inline vec<real> logspace(real start, real stop, idx n, real base = 10.0) {
    vec<real> out = linspace(start, stop, n);
    for (real &exponent : out) {
        exponent = std::pow(base, exponent);
    }
    return out;
}

/// @brief Return the integers \f$\mathrm{start}, \ldots, \mathrm{start} + n - 1\f$.
inline array<int> int_range(int start, int n) {
    assert(n >= 0);
    array<int> out(static_cast<idx>(n));
    std::iota(out.begin(), out.end(), start);
    return out;
}

// Random number generation (wrapping the mt19937 boilerplate)

/// @brief Random number generator (Mersenne Twister) with a seed.
struct rng_state {
    std::mt19937 engine;

    explicit rng_state(uint32_t seed) : engine(seed) {}

    /// @brief Seed the generator from hardware entropy.
    rng_state() : engine(std::random_device{}()) {}
};

/// @brief Draw a uniform sample from \f$[\mathrm{lo}, \mathrm{hi})\f$.
inline real rng_uniform(rng_state *r, real lo, real hi) {
    return std::uniform_real_distribution<real>{lo, hi}(r->engine);
}

/// @brief Draw a normal sample: \f$\mathcal{N}(\mu, \sigma^2)\f$.
inline real rng_normal(rng_state *r, real mean, real stddev) {
    return std::normal_distribution<real>{mean, stddev}(r->engine);
}

/// @brief Draw a uniform integer from \f$[\mathrm{lo}, \mathrm{hi}]\f$.
inline int rng_int(rng_state *r, int lo, int hi) {
    return std::uniform_int_distribution<int>{lo, hi}(r->engine);
}

// Spatial distributions

/// @brief Evaluate a 2D Gaussian centred at \f$(c_x, c_y)\f$:
/// \f$\exp(-((x - c_x)^2 + (y - c_y)^2) / 2\sigma^2)\f$.
inline real gaussian2d(real x, real y, real cx, real cy, real sigma) {
    real dx = x - cx, dy = y - cy;
    return std::exp(-((dx * dx) + (dy * dy)) / (2.0 * sigma * sigma));
}

} // namespace num
