/// @file roots/roots.hpp
/// @brief Pure template root-finding methods for scalar equations \f$f(x^*) = 0\f$.
#pragma once

#include "core/types.hpp"
#include <cmath>
#include <concepts>
#include <stdexcept>

namespace num {

template <typename Float = double>
struct BasicRootResult {
    Float root{};
    idx iterations{};
    Float residual{}; ///< Absolute residual \f$|f(x^*)||\f$
    bool converged{};
};

using RootResult = BasicRootResult<real>;

/// @brief Bisection method for finding a root on a bracketing interval \f$[a, b]\f$ where \f$f(a) f(b) < 0\f$.
template <typename Float = double, std::invocable<Float> Func = ScalarFn>
inline BasicRootResult<Float> bisection(Func &&f, Float a, Float b,
                                       Float tol = Float{1e-10}, idx max_iter = 1000) {
    Float fa = f(a), fb = f(b);
    if (fa * fb > Float{0}) {
        throw std::invalid_argument("bisection: f(a) and f(b) must have opposite signs");
    }

    for (idx i = 0; i < max_iter; ++i) {
        Float mid = Float{0.5} * (a + b);
        Float fm = f(mid);
        if (std::abs(fm) < tol || Float{0.5} * (b - a) < tol) {
            return {mid, i + 1, std::abs(fm), true};
        }
        if (fa * fm < Float{0}) {
            b = mid;
            fb = fm;
        } else {
            a = mid;
            fa = fm;
        }
    }
    Float mid = Float{0.5} * (a + b);
    return {mid, max_iter, std::abs(f(mid)), false};
}

/// @brief Newton–Raphson method with quadratic convergence using analytical derivative \f$f'(x)\f$.
template <typename Float = double, std::invocable<Float> Func = ScalarFn,
          std::invocable<Float> DFunc = ScalarFn>
inline BasicRootResult<Float> newton(Func &&f, DFunc &&df, Float x0,
                                     Float tol = Float{1e-10}, idx max_iter = 1000) {
    Float x = x0;
    for (idx i = 0; i < max_iter; ++i) {
        Float fx = f(x);
        if (std::abs(fx) < tol) {
            return {x, i + 1, std::abs(fx), true};
        }
        Float dfx = df(x);
        if (std::abs(dfx) < Float{1e-14}) {
            return {x, i + 1, std::abs(fx), false}; // near-zero derivative
        }
        x -= fx / dfx;
    }
    return {x, max_iter, std::abs(f(x)), false};
}

/// @brief Secant quasi-Newton method requiring two initial guesses \f$x_0, x_1\f$.
template <typename Float = double, std::invocable<Float> Func = ScalarFn>
inline BasicRootResult<Float> secant(Func &&f, Float x0, Float x1,
                                     Float tol = Float{1e-10}, idx max_iter = 1000) {
    Float f0 = f(x0), f1 = f(x1);
    for (idx i = 0; i < max_iter; ++i) {
        if (std::abs(f1) < tol) {
            return {x1, i + 1, std::abs(f1), true};
        }
        Float df = f1 - f0;
        if (std::abs(df) < Float{1e-14}) {
            return {x1, i + 1, std::abs(f1), false}; // stagnation
        }
        Float x2 = x1 - (f1 * (x1 - x0) / df);
        x0 = x1;
        f0 = f1;
        x1 = x2;
        f1 = f(x1);
    }
    return {x1, max_iter, std::abs(f1), false};
}

/// @brief Brent's hybrid root solver (bisection + secant + inverse quadratic interpolation).
template <typename Float = double, std::invocable<Float> Func = ScalarFn>
inline BasicRootResult<Float> brent(Func &&f, Float a, Float b,
                                    Float tol = Float{1e-10}, idx max_iter = 1000) {
    Float fa = f(a), fb = f(b);
    if (fa * fb > Float{0}) {
        throw std::invalid_argument("brent: f(a) and f(b) must have opposite signs");
    }

    Float c = a, fc = fa;
    Float d = b - a, e = d;

    for (idx i = 0; i < max_iter; ++i) {
        if (fb * fc > Float{0}) {
            c = a;
            fc = fa;
            d = e = b - a;
        }
        if (std::abs(fc) < std::abs(fb)) {
            a = b;
            fa = fb;
            b = c;
            fb = fc;
            c = a;
            fc = fa;
        }

        Float tol1 = (Float{2.0} * Float{1e-15} * std::abs(b)) + (Float{0.5} * tol);
        Float mid = Float{0.5} * (c - b);

        if (std::abs(mid) <= tol1 || std::abs(fb) < tol) {
            return {b, i + 1, std::abs(fb), true};
        }

        if (std::abs(e) >= tol1 && std::abs(fa) > std::abs(fb)) {
            Float s = fb / fa;
            Float p{}, q{};
            if (a == c) {
                p = Float{2.0} * mid * s;
                q = Float{1.0} - s;
            } else {
                Float r = fb / fc;
                Float t = fa / fc;
                p = s * ((Float{2.0} * mid * t * (t - r)) - ((b - a) * (r - Float{1.0})));
                q = (t - Float{1.0}) * (r - Float{1.0}) * (s - Float{1.0});
            }
            if (p > Float{0}) {
                q = -q;
            } else {
                p = -p;
            }

            Float e_prev = e;
            if (Float{2.0} * p < std::min((Float{3.0} * mid * q) - std::abs(tol1 * q), std::abs(e_prev * q))) {
                e = d;
                d = p / q;
            } else {
                d = mid;
                e = mid;
            }
        } else {
            d = mid;
            e = mid;
        }

        a = b;
        fa = fb;
        b += (std::abs(d) > tol1) ? d : (mid > Float{0} ? tol1 : -tol1);
        fb = f(b);
    }
    return {b, max_iter, std::abs(fb), false};
}

} // namespace num
