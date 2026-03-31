/// @file roots.cpp
#include "analysis/roots.hpp"
#include <cmath>
#include <stdexcept>

namespace num {

RootResult bisection(ScalarFn f, real a, real b, real tol, idx max_iter) {
    real fa = f(a), fb = f(b);
    if (fa * fb > 0.0)
        throw std::invalid_argument(
            "bisection: f(a) and f(b) must have opposite signs");

    for (idx i = 0; i < max_iter; ++i) {
        real mid = 0.5 * (a + b);
        real fm  = f(mid);
        if (std::abs(fm) < tol || 0.5 * (b - a) < tol)
            return {mid, i + 1, std::abs(fm), true};
        if (fa * fm < 0.0) {
            b  = mid;
            fb = fm;
        } else {
            a  = mid;
            fa = fm;
        }
    }
    real mid = 0.5 * (a + b);
    return {mid, max_iter, std::abs(f(mid)), false};
}

RootResult newton(ScalarFn f, ScalarFn df, real x0, real tol, idx max_iter) {
    real x = x0;
    for (idx i = 0; i < max_iter; ++i) {
        real fx = f(x);
        if (std::abs(fx) < tol)
            return {x, i + 1, std::abs(fx), true};
        real dfx = df(x);
        if (std::abs(dfx) < 1e-14)
            return {x, i + 1, std::abs(fx), false}; // near-zero derivative
        x -= fx / dfx;
    }
    return {x, max_iter, std::abs(f(x)), false};
}

RootResult secant(ScalarFn f, real x0, real x1, real tol, idx max_iter) {
    real f0 = f(x0), f1 = f(x1);
    for (idx i = 0; i < max_iter; ++i) {
        if (std::abs(f1) < tol)
            return {x1, i + 1, std::abs(f1), true};
        real df = f1 - f0;
        if (std::abs(df) < 1e-14)
            return {x1, i + 1, std::abs(f1), false}; // stagnation
        real x2 = x1 - f1 * (x1 - x0) / df;
        x0      = x1;
        f0      = f1;
        x1      = x2;
        f1      = f(x1);
    }
    return {x1, max_iter, std::abs(f1), false};
}

RootResult brent(ScalarFn f, real a, real b, real tol, idx max_iter) {
    real fa = f(a), fb = f(b);
    if (fa * fb > 0.0)
        throw std::invalid_argument(
            "brent: f(a) and f(b) must have opposite signs");

    // c is the point such that [b,c] is always a valid bracket
    real c = a, fc = fa;
    real d = b - a, e = d;

    for (idx i = 0; i < max_iter; ++i) {
        // ensure b is the best estimate
        if (fb * fc > 0.0) {
            c  = a;
            fc = fa;
            d = e = b - a;
        }
        if (std::abs(fc) < std::abs(fb)) {
            a  = b;
            fa = fb;
            b  = c;
            fb = fc;
            c  = a;
            fc = fa;
        }

        real tol1 = 2.0 * 1e-15 * std::abs(b) + 0.5 * tol;
        real mid  = 0.5 * (c - b);

        if (std::abs(mid) <= tol1 || std::abs(fb) < tol)
            return {b, i + 1, std::abs(fb), true};

        if (std::abs(e) >= tol1 && std::abs(fa) > std::abs(fb)) {
            // attempt interpolation
            real s = fb / fa;
            real p, q;
            if (a == c) {
                // secant step
                p = 2.0 * mid * s;
                q = 1.0 - s;
            } else {
                // inverse quadratic interpolation
                real r = fb / fc;
                real t = fa / fc;
                p      = s * (2.0 * mid * t * (t - r) - (b - a) * (r - 1.0));
                q      = (t - 1.0) * (r - 1.0) * (s - 1.0);
            }
            if (p > 0.0)
                q = -q;
            else
                p = -p;

            // accept interpolation only if step is smaller than previous
            real e_prev = e;
            if (2.0 * p < std::min(3.0 * mid * q - std::abs(tol1 * q),
                                   std::abs(e_prev * q))) {
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

        a  = b;
        fa = fb;
        b += (std::abs(d) > tol1) ? d : (mid > 0.0 ? tol1 : -tol1);
        fb = f(b);
    }
    return {b, max_iter, std::abs(fb), false};
}

} // namespace num
