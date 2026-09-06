/// @file quadrature/quadrature.hpp
/// @brief Pure template numerical integration (quadrature) on [a, b].
#pragma once

#include "core/policy.hpp"
#include "core/types.hpp"
#include <cmath>
#include <concepts>
#include <stdexcept>
#include <vector>

#ifdef NUMERICS_HAS_OMP
#ifdef _OPENMP
#include <omp.h>
#endif
#endif

namespace num {

namespace detail {

// Gauss-Legendre nodes and weights on [-1, 1] for p = 1..5
static constexpr double GL_NODES[5][5] = {
    {0.0, 0, 0, 0, 0},
    {-0.5773502691896257, 0.5773502691896257, 0, 0, 0},
    {-0.7745966692414834, 0.0, 0.7745966692414834, 0, 0},
    {-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526, 0},
    {-0.9061798459386640, -0.5384693101056831, 0.0, 0.5384693101056831, 0.9061798459386640}};

static constexpr double GL_WEIGHTS[5][5] = {
    {2.0, 0, 0, 0, 0},
    {1.0, 1.0, 0, 0, 0},
    {0.5555555555555556, 0.8888888888888889, 0.5555555555555556, 0, 0},
    {0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538, 0},
    {0.2369268850561891, 0.4786286704993665, 0.5688888888888889, 0.4786286704993665,
     0.2369268850561891}};

template <typename Float, typename Func>
inline Float adaptive_helper(Func &&f, Float a, Float b, Float fa, Float fm, Float fb, Float whole,
                             Float tol, idx depth) {
    Float mid = Float{0.5} * (a + b);
    Float fl = f(Float{0.5} * (a + mid));
    Float fr = f(Float{0.5} * (mid + b));

    Float left = (b - a) / Float{12.0} * (fa + (Float{4.0} * fl) + fm);
    Float right = (b - a) / Float{12.0} * (fm + (Float{4.0} * fr) + fb);
    Float delta = left + right - whole;

    if (depth == 0 || std::abs(delta) <= Float{15.0} * tol) {
        return left + right + (delta / Float{15.0});
    }

    return adaptive_helper(f, a, mid, fa, fl, fm, left, tol * Float{0.5}, depth - 1) +
           adaptive_helper(f, mid, b, fm, fr, fb, right, tol * Float{0.5}, depth - 1);
}

} // namespace detail

/// @brief Composite Trapezoidal numerical integration over \f$[a, b]\f$ with \f$n\f$ panels.
///
/// Computes \f$\int_a^b f(x)\,\mathrm{d}x \approx h \left( \frac{f(a)+f(b)}{2} + \sum_{i=1}^{n-1} f(a+ih) \right)\f$ with \f$\mathcal{O}(h^2)\f$ error.
///
/// @tparam Float Floating point scalar type (`double`, `float`).
/// @tparam Func Integrand callable `Float(Float x)`.
/// @param f Integrand function.
/// @param a Lower integration limit.
/// @param b Upper integration limit.
/// @param n Number of uniform subintervals / panels (default: 100).
/// @tparam Parallel Thread the panel sum with OpenMP (default: sequential).
/// @return Approximated integral value.
/// @see simpson, gauss_legendre, romberg
template <bool Parallel = false, typename Float = double, std::invocable<Float> Func = scalar_fn>
inline Float trapz(Func &&f, Float a, Float b, idx n = 100) {
    Float h = (b - a) / static_cast<Float>(n);
    Float sum = Float{0};
#if defined(NUMERICS_HAS_OMP)
    if constexpr (Parallel) {
#pragma omp parallel for reduction(+ : sum) schedule(static)
        for (idx i = 1; i < n; ++i) {
            sum += f(a + (static_cast<Float>(i) * h));
        }
    } else
#endif
    {
        for (idx i = 1; i < n; ++i) {
            sum += f(a + (static_cast<Float>(i) * h));
        }
    }
    return h * ((Float{0.5} * (f(a) + f(b))) + sum);
}

/// @brief Composite Simpson's 1/3 numerical integration with \f$n\f$ panels (\f$n\f$ even).
///
/// Approximates \f$\int_a^b f(x)\,\mathrm{d}x\f$ using piecewise quadratic polynomials, achieving \f$\mathcal{O}(h^4)\f$ error.
///
/// @tparam Float Floating point scalar type.
/// @tparam Func Integrand callable `Float(Float x)`.
/// @param f Integrand function.
/// @param a Lower limit.
/// @param b Upper limit.
/// @param n Number of panels (must be even, default: 100).
/// @tparam Parallel Thread the panel sum with OpenMP (default: sequential).
/// @return Approximated integral value.
/// @throws std::invalid_argument If `n` is not even.
/// @see trapz, adaptive_simpson, gauss_legendre
template <bool Parallel = false, typename Float = double, std::invocable<Float> Func = scalar_fn>
inline Float simpson(Func &&f, Float a, Float b, idx n = 100) {
    if (n % 2 != 0) {
        throw std::invalid_argument("simpson: n must be even");
    }
    Float h = (b - a) / static_cast<Float>(n);
    Float sum = f(a) + f(b);
#if defined(NUMERICS_HAS_OMP)
    if constexpr (Parallel) {
#pragma omp parallel for reduction(+ : sum) schedule(static)
        for (idx i = 1; i < n; ++i) {
            sum += f(a + (static_cast<Float>(i) * h)) * (i % 2 == 0 ? Float{2.0} : Float{4.0});
        }
    } else
#endif
    {
        for (idx i = 1; i < n; ++i) {
            sum += f(a + (static_cast<Float>(i) * h)) * (i % 2 == 0 ? Float{2.0} : Float{4.0});
        }
    }
    return (h / Float{3.0}) * sum;
}

/// @brief Gauss–Legendre quadrature with \f$p \in [1, 5]\f$ optimal nodes.
///
/// Exact for all polynomials up to degree \f$2p - 1\f$ using precomputed roots and weights of Legendre polynomials.
///
/// @tparam Float Floating point scalar type.
/// @tparam Func Integrand callable.
/// @param f Integrand function.
/// @param a Lower integration limit.
/// @param b Upper integration limit.
/// @param p Number of quadrature nodes (\f$1 \le p \le 5\f$, default: 5).
/// @return Exact or high-precision approximation of the integral.
/// @throws std::invalid_argument If `p` is not in range \f$[1, 5]\f$.
template <typename Float = double, std::invocable<Float> Func = scalar_fn>
inline Float gauss_legendre(Func &&f, Float a, Float b, idx p = 5) {
    if (p < 1 || p > 5) {
        throw std::invalid_argument("gauss_legendre: p must be 1..5");
    }
    Float mid = Float{0.5} * (a + b);
    Float half = Float{0.5} * (b - a);
    Float sum = Float{0};
    for (idx i = 0; i < p; ++i) {
        sum += static_cast<Float>(detail::GL_WEIGHTS[p - 1][i]) *
               f(mid + (half * static_cast<Float>(detail::GL_NODES[p - 1][i])));
    }
    return half * sum;
}

/// @brief Adaptive Simpson quadrature with recursive interval bisection and error control.
///
/// Automatically concentrates function evaluations in regions of high curvature.
///
/// @tparam Float Floating point scalar type.
/// @tparam Func Integrand callable.
/// @param f Integrand function.
/// @param a Lower limit.
/// @param b Upper limit.
/// @param tol Error tolerance for adaptive refinement (default: 1e-8).
/// @param max_depth Maximum recursion depth before termination (default: 50).
/// @return High-accuracy integral approximation satisfying error tolerance.
template <typename Float = double, std::invocable<Float> Func = scalar_fn>
inline Float adaptive_simpson(Func &&f, Float a, Float b, Float tol = Float{1e-8}, idx max_depth = 50) {
    Float fa = f(a), fb = f(b), fm = f(Float{0.5} * (a + b));
    Float est = (b - a) / Float{6.0} * (fa + (Float{4.0} * fm) + fb);
    return detail::adaptive_helper(f, a, b, fa, fm, fb, est, tol, max_depth);
}

/// @brief Romberg numerical integration using Richardson extrapolation on trapezoidal sums.
///
/// Generates triangular tableau \f$R(j, k)\f$ with successive error cancellations of order \f$\mathcal{O}(h^{2(k+1)})\f$.
///
/// @tparam Float Floating point scalar type.
/// @tparam Func Integrand callable.
/// @param f Integrand function.
/// @param a Lower limit.
/// @param b Upper limit.
/// @param tol Convergence tolerance on diagonal tableau entries (default: 1e-10).
/// @param max_levels Maximum tableau extrapolation levels (default: 12).
/// @return Extrapolated integral approximation.
template <typename Float = double, std::invocable<Float> Func = scalar_fn>
inline Float romberg(Func &&f, Float a, Float b, Float tol = Float{1e-10}, idx max_levels = 12) {
    std::vector<std::vector<Float>> R(max_levels, std::vector<Float>(max_levels, Float{0}));
    R[0][0] = Float{0.5} * (b - a) * (f(a) + f(b));

    for (idx i = 1; i < max_levels; ++i) {
        idx n = idx(1) << i;
        Float h = (b - a) / static_cast<Float>(n);
        Float sum = Float{0};
        for (idx k = 1; k < n; k += 2) {
            sum += f(a + (static_cast<Float>(k) * h));
        }
        R[i][0] = (Float{0.5} * R[i - 1][0]) + (h * sum);

        Float factor = Float{1.0};
        for (idx j = 1; j <= i; ++j) {
            factor *= Float{4.0};
            R[i][j] = R[i][j - 1] + ((R[i][j - 1] - R[i - 1][j - 1]) / (factor - Float{1.0}));
        }

        if (i > 0 && std::abs(R[i][i] - R[i - 1][i - 1]) < tol) {
            return R[i][i];
        }
    }
    return R[max_levels - 1][max_levels - 1];
}

} // namespace num
