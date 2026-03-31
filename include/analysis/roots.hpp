/// @file roots.hpp
/// @brief Root-finding methods for scalar equations f(x) = 0
#pragma once

#include "analysis/types.hpp"

namespace num {

struct RootResult {
    real root;
    idx  iterations;
    real residual; ///< |f(root)|
    bool converged;
};

/// @brief Bisection method
/// @param f   Continuous function
/// @param a,b Bracket: f(a) and f(b) must have opposite signs
/// @param tol Convergence tolerance on |f(root)| and interval width
/// @param max_iter Maximum iterations
RootResult bisection(ScalarFn f,
                     real     a,
                     real     b,
                     real     tol      = 1e-10,
                     idx      max_iter = 1000);

/// @brief Newton-Raphson method
/// @param f   Function
/// @param df  Derivative of f
/// @param x0  Initial guess
/// @param tol Convergence tolerance
/// @param max_iter Maximum iterations
RootResult newton(ScalarFn f,
                  ScalarFn df,
                  real     x0,
                  real     tol      = 1e-10,
                  idx      max_iter = 1000);

/// @brief Secant method (Newton without derivative)
/// @param f     Function
/// @param x0,x1 Two distinct initial guesses
/// @param tol   Convergence tolerance
/// @param max_iter Maximum iterations
RootResult secant(ScalarFn f,
                  real     x0,
                  real     x1,
                  real     tol      = 1e-10,
                  idx      max_iter = 1000);

/// @brief Brent's method (bisection + secant + inverse quadratic interpolation)
///
/// Guaranteed to converge if f(a) and f(b) have opposite signs.
/// Superlinear convergence near smooth roots.
/// Preferred method when a reliable bracket is available.
///
/// @param f     Function
/// @param a,b   Bracket: f(a) and f(b) must have opposite signs
/// @param tol   Convergence tolerance
/// @param max_iter Maximum iterations
RootResult brent(ScalarFn f,
                 real     a,
                 real     b,
                 real     tol      = 1e-10,
                 idx      max_iter = 1000);

} // namespace num
