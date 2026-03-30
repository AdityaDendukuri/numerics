#include "ode/ode.hpp"
#include <algorithm>
#include <cmath>

namespace num {

// helpers

static void axpy_vec(real alpha, const Vector& x, Vector& y) {
    for (idx i = 0; i < x.size(); ++i) y[i] += alpha * x[i];
}

// Euler

ODEResult ode_euler(ODERhsFn f, Vector y0, real t0, real t1, real h,
                    StepCallback on_step) {
    const idx n = y0.size();
    Vector dydt(n);
    idx steps = 0;

    for (real t = t0; t < t1 - 1e-14 * std::abs(t1); t += h) {
        real dt = std::min(h, t1 - t);
        f(t, y0, dydt);
        axpy_vec(dt, dydt, y0);
        ++steps;
        if (on_step) on_step(t + dt, y0);
    }
    return {std::move(y0), t1, steps, true};
}

// RK4

ODEResult ode_rk4(ODERhsFn f, Vector y0, real t0, real t1, real h,
                  StepCallback on_step) {
    const idx n = y0.size();
    Vector k1(n), k2(n), k3(n), k4(n), ytmp(n);
    idx steps = 0;

    for (real t = t0; t < t1 - 1e-14 * std::abs(t1); t += h) {
        real dt = std::min(h, t1 - t);

        f(t, y0, k1);

        for (idx i = 0; i < n; ++i) ytmp[i] = y0[i] + 0.5 * dt * k1[i];
        f(t + 0.5 * dt, ytmp, k2);

        for (idx i = 0; i < n; ++i) ytmp[i] = y0[i] + 0.5 * dt * k2[i];
        f(t + 0.5 * dt, ytmp, k3);

        for (idx i = 0; i < n; ++i) ytmp[i] = y0[i] + dt * k3[i];
        f(t + dt, ytmp, k4);

        for (idx i = 0; i < n; ++i)
            y0[i] += (dt / 6.0) * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]);
        ++steps;
        if (on_step) on_step(t + dt, y0);
    }
    return {std::move(y0), t1, steps, true};
}

// RK45 (Dormand-Prince)
//
// Butcher tableau from Dormand & Prince, "A family of embedded Runge-Kutta
// formulae", J. Comput. Appl. Math. 6(1), 19-26 (1980).
// 5th-order propagation, 4th-order embedded error estimate, FSAL property.

ODEResult ode_rk45(ODERhsFn f, Vector y0, real t0, real t1,
                   real rtol, real atol, real h0, idx max_steps,
                   StepCallback on_step) {
    static constexpr real a21 = 1.0/5.0;
    static constexpr real a31 = 3.0/40.0,     a32 = 9.0/40.0;
    static constexpr real a41 = 44.0/45.0,    a42 = -56.0/15.0,    a43 = 32.0/9.0;
    static constexpr real a51 = 19372.0/6561.0, a52 = -25360.0/2187.0,
                          a53 = 64448.0/6561.0, a54 = -212.0/729.0;
    static constexpr real a61 = 9017.0/3168.0, a62 = -355.0/33.0,
                          a63 = 46732.0/5247.0, a64 = 49.0/176.0,
                          a65 = -5103.0/18656.0;

    static constexpr real b1 = 35.0/384.0,  b3 = 500.0/1113.0,
                          b4 = 125.0/192.0, b5 = -2187.0/6784.0, b6 = 11.0/84.0;

    static constexpr real e1 =  71.0/57600.0,  e3 = -71.0/16695.0,
                          e4 =  71.0/1920.0,   e5 = -17253.0/339200.0,
                          e6 =  22.0/525.0,    e7 = -1.0/40.0;

    const idx n = y0.size();
    Vector k1(n), k2(n), k3(n), k4(n), k5(n), k6(n), k7(n), ytmp(n), err(n);

    real h = std::min(h0, t1 - t0);
    real t = t0;
    idx steps = 0;
    bool converged = true;

    f(t, y0, k1);

    while (t < t1 - 1e-14 * std::abs(t1)) {
        if (steps >= max_steps) { converged = false; break; }
        h = std::min(h, t1 - t);

        for (idx i = 0; i < n; ++i) ytmp[i] = y0[i] + h * a21 * k1[i];
        f(t + h/5.0, ytmp, k2);

        for (idx i = 0; i < n; ++i) ytmp[i] = y0[i] + h*(a31*k1[i] + a32*k2[i]);
        f(t + 3*h/10.0, ytmp, k3);

        for (idx i = 0; i < n; ++i) ytmp[i] = y0[i] + h*(a41*k1[i] + a42*k2[i] + a43*k3[i]);
        f(t + 4*h/5.0, ytmp, k4);

        for (idx i = 0; i < n; ++i)
            ytmp[i] = y0[i] + h*(a51*k1[i] + a52*k2[i] + a53*k3[i] + a54*k4[i]);
        f(t + 8*h/9.0, ytmp, k5);

        for (idx i = 0; i < n; ++i)
            ytmp[i] = y0[i] + h*(a61*k1[i] + a62*k2[i] + a63*k3[i] + a64*k4[i] + a65*k5[i]);
        f(t + h, ytmp, k6);

        for (idx i = 0; i < n; ++i)
            ytmp[i] = y0[i] + h*(b1*k1[i] + b3*k3[i] + b4*k4[i] + b5*k5[i] + b6*k6[i]);

        f(t + h, ytmp, k7);

        for (idx i = 0; i < n; ++i)
            err[i] = h * (e1*k1[i] + e3*k3[i] + e4*k4[i] + e5*k5[i] + e6*k6[i] + e7*k7[i]);

        real err_norm = 0;
        for (idx i = 0; i < n; ++i) {
            real sc = atol + rtol * std::max(std::abs(y0[i]), std::abs(ytmp[i]));
            real ratio = err[i] / sc;
            err_norm = std::max(err_norm, std::abs(ratio));
        }

        if (err_norm <= 1.0) {
            t += h;
            y0 = ytmp;
            k1 = k7;
            ++steps;
            if (on_step) on_step(t, y0);
        }

        real factor = 0.9 * std::pow(err_norm + 1e-10, -0.2);
        factor = std::max(real(0.1), std::min(real(10.0), factor));
        h *= factor;
    }

    return {std::move(y0), t, steps, converged};
}

// Velocity Verlet -- KDK form with FSAL; symplectic, time-reversible.

SymplecticResult ode_verlet(AccelFn accel, Vector q0, Vector v0,
                             real t0, real t1, real h,
                             SymplecticCallback on_step) {
    const idx n = q0.size();
    Vector a_cur(n), a_next(n);
    accel(q0, a_cur);

    idx steps = 0;
    real t = t0;

    while (t < t1 - 1e-14 * std::abs(t1)) {
        real dt = std::min(h, t1 - t);

        // q_{n+1} = q_n + h*v_n + h^2/2 * a_n
        for (idx i = 0; i < n; ++i)
            q0[i] += dt * v0[i] + 0.5 * dt * dt * a_cur[i];

        accel(q0, a_next);

        // v_{n+1} = v_n + h/2 * (a_n + a_{n+1})
        for (idx i = 0; i < n; ++i)
            v0[i] += 0.5 * dt * (a_cur[i] + a_next[i]);

        std::swap(a_cur, a_next);
        t += dt;
        ++steps;
        if (on_step) on_step(t, q0, v0);
    }

    return {std::move(q0), std::move(v0), t, steps};
}

// Yoshida 4th-order symplectic (Yoshida 1990, Phys. Lett. A 150:262).

SymplecticResult ode_yoshida4(AccelFn accel, Vector q0, Vector v0,
                               real t0, real t1, real h,
                               SymplecticCallback on_step) {
    static const real w1 = 1.0 / (2.0 - std::cbrt(2.0));
    static const real w0 = 1.0 - 2.0 * w1;
    static const real c1 = w1 * 0.5;
    static const real c2 = (w0 + w1) * 0.5;
    // c3 == c2, c4 == c1
    static const real d1 = w1;
    static const real d2 = w0;
    // d3 == d1

    const idx n = q0.size();
    Vector acc(n);
    idx steps = 0;
    real t = t0;

    while (t < t1 - 1e-14 * std::abs(t1)) {
        real dt = std::min(h, t1 - t);

        // drift c1
        for (idx i = 0; i < n; ++i) q0[i] += c1 * dt * v0[i];
        // kick d1
        accel(q0, acc);
        for (idx i = 0; i < n; ++i) v0[i] += d1 * dt * acc[i];
        // drift c2
        for (idx i = 0; i < n; ++i) q0[i] += c2 * dt * v0[i];
        // kick d2
        accel(q0, acc);
        for (idx i = 0; i < n; ++i) v0[i] += d2 * dt * acc[i];
        // drift c3 (== c2)
        for (idx i = 0; i < n; ++i) q0[i] += c2 * dt * v0[i];
        // kick d3 (== d1)
        accel(q0, acc);
        for (idx i = 0; i < n; ++i) v0[i] += d1 * dt * acc[i];
        // drift c4 (== c1)
        for (idx i = 0; i < n; ++i) q0[i] += c1 * dt * v0[i];

        t += dt;
        ++steps;
        if (on_step) on_step(t, q0, v0);
    }

    return {std::move(q0), std::move(v0), t, steps};
}

// RK4 for second-order systems (Nystrom form)
// 4 force evaluations per step; O(h^4) local truncation error; not symplectic.

SymplecticResult ode_rk4_2nd(AccelFn accel, Vector q0, Vector v0,
                              real t0, real t1, real h,
                              SymplecticCallback on_step) {
    const idx n = q0.size();
    Vector a1(n), a2(n), a3(n), a4(n), qtmp(n);
    idx steps = 0;
    real t = t0;

    while (t < t1 - 1e-14 * std::abs(t1)) {
        real dt = std::min(h, t1 - t);

        accel(q0, a1);

        for (idx i = 0; i < n; ++i) qtmp[i] = q0[i] + 0.5 * dt * v0[i];
        accel(qtmp, a2);

        for (idx i = 0; i < n; ++i) qtmp[i] = q0[i] + 0.5 * dt * (v0[i] + 0.5 * dt * a1[i]);
        accel(qtmp, a3);

        for (idx i = 0; i < n; ++i) qtmp[i] = q0[i] + dt * (v0[i] + 0.5 * dt * a2[i]);
        accel(qtmp, a4);

        for (idx i = 0; i < n; ++i) {
            q0[i] += dt * v0[i] + (dt * dt / 6.0) * (a1[i] + a2[i] + a3[i]);
            v0[i] += (dt / 6.0) * (a1[i] + 2.0 * a2[i] + 2.0 * a3[i] + a4[i]);
        }

        t += dt;
        ++steps;
        if (on_step) on_step(t, q0, v0);
    }

    return {std::move(q0), std::move(v0), t, steps};
}

} // namespace num
