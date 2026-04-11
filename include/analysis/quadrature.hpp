/// @file quadrature.hpp
/// @brief Numerical integration (quadrature) on [a, b]
#pragma once

#include "analysis/types.hpp"
#include "core/policy.hpp"

namespace num {

/// @brief Trapezoidal rule with n panels
/// @param backend  Backend::omp parallelises the panel sum
real trapz(ScalarFn f, real a, real b, idx n = 100,
           Backend backend = Backend::seq);

/// @brief Simpson's 1/3 rule with n panels (n must be even)
/// @param backend  Backend::omp parallelises the panel sum
real simpson(ScalarFn f, real a, real b, idx n = 100,
             Backend backend = Backend::seq);

/// @brief Gauss-Legendre quadrature (exact for polynomials up to degree 2p-1)
/// @param f Integrand
/// @param a Lower bound
/// @param b Upper bound
/// @param p Number of quadrature points (1 to 5 supported)
real gauss_legendre(ScalarFn f, real a, real b, idx p = 5);

/// @brief Adaptive Simpson quadrature
/// @param f Integrand
/// @param a Lower bound
/// @param b Upper bound
/// @param tol Absolute error tolerance
/// @param max_depth Maximum recursion depth
real adaptive_simpson(ScalarFn f, real a, real b, real tol = 1e-8,
                      idx max_depth = 50);

/// @brief Romberg integration (Richardson extrapolation on trapezoidal rule)
/// @param f Integrand
/// @param a Lower bound
/// @param b Upper bound
/// @param tol Convergence tolerance
/// @param max_levels Maximum refinement levels
real romberg(ScalarFn f, real a, real b, real tol = 1e-10, idx max_levels = 12);

} // namespace num
