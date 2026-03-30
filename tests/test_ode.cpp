/// @file test_ode.cpp
/// @brief Tests for ODE integrators: convergence, energy conservation, callbacks.
#include <gtest/gtest.h>
#include "ode/ode.hpp"
#include <cmath>
#include <vector>

using namespace num;

// ── Helpers ───────────────────────────────────────────────────────────────────

// Harmonic oscillator: y'' = -y  →  y=[x,v], dy/dt=[v,-x]
// Exact: x(t) = x0*cos(t) + v0*sin(t)
static ODERhsFn harmonic_osc() {
    return [](real t, const Vector& y, Vector& dy) {
        (void)t;
        dy[0] =  y[1];
        dy[1] = -y[0];
    };
}

// Kepler acceleration for 2D orbit (G=1, central mass=1 at origin)
// q = [x, y], a = -q / |q|^3
static AccelFn kepler_accel() {
    return [](const Vector& q, Vector& a) {
        real r2 = q[0]*q[0] + q[1]*q[1];
        real r3 = r2 * std::sqrt(r2);
        a[0] = -q[0] / r3;
        a[1] = -q[1] / r3;
    };
}

// Total energy for Kepler: E = 0.5*(vx²+vy²) - 1/r
static real kepler_energy(const Vector& q, const Vector& v) {
    real KE = 0.5 * (v[0]*v[0] + v[1]*v[1]);
    real r  = std::sqrt(q[0]*q[0] + q[1]*q[1]);
    return KE - 1.0 / r;
}

// Circular orbit at radius r=1: q=(1,0), v=(0,1), E=-0.5
static Vector kepler_q0() { return {1.0, 0.0}; }
static Vector kepler_v0() { return {0.0, 1.0}; }

// ── Euler convergence ─────────────────────────────────────────────────────────

TEST(ODE_Euler, OrderOne) {
    // y' = -y, y(0)=1, exact y(t)=exp(-t)
    auto f = [](real t, const Vector& y, Vector& dy) { (void)t; dy[0] = -y[0]; };

    auto err = [&](real h) {
        auto res = ode_euler(f, {1.0}, 0.0, 1.0, h);
        return std::abs(res.y[0] - std::exp(-1.0));
    };

    real e1 = err(0.01), e2 = err(0.005);
    real rate = std::log(e1 / e2) / std::log(2.0);
    EXPECT_NEAR(rate, 1.0, 0.05);
}

// ── RK4 convergence ───────────────────────────────────────────────────────────

TEST(ODE_RK4, OrderFour) {
    auto f = harmonic_osc();
    Vector y0{1.0, 0.0};  // x0=1, v0=0 → x(t)=cos(t)

    auto err = [&](real h) {
        auto res = ode_rk4(f, y0, 0.0, 1.0, h);
        return std::abs(res.y[0] - std::cos(1.0));
    };

    real e1 = err(0.1), e2 = err(0.05);
    real rate = std::log(e1 / e2) / std::log(2.0);
    EXPECT_NEAR(rate, 4.0, 0.1);
}

TEST(ODE_RK4, HarmonicOscillatorAccuracy) {
    auto res = ode_rk4(harmonic_osc(), {1.0, 0.0}, 0.0, 2.0 * M_PI, 0.01);
    // After one full period: x ≈ 1, v ≈ 0
    EXPECT_NEAR(res.y[0], 1.0, 1e-8);
    EXPECT_NEAR(res.y[1], 0.0, 1e-8);
}

// ── RK45 adaptive ─────────────────────────────────────────────────────────────

TEST(ODE_RK45, HarmonicOscillator) {
    auto res = ode_rk45(harmonic_osc(), {1.0, 0.0}, 0.0, 2.0 * M_PI,
                        1e-9, 1e-12);
    EXPECT_TRUE(res.converged);
    EXPECT_NEAR(res.y[0], 1.0, 1e-7);
    EXPECT_NEAR(res.y[1], 0.0, 1e-7);
}

TEST(ODE_RK45, ExponentialDecay) {
    auto f = [](real t, const Vector& y, Vector& dy) { (void)t; dy[0] = -y[0]; };
    auto res = ode_rk45(f, {1.0}, 0.0, 5.0, 1e-10, 1e-12);
    EXPECT_NEAR(res.y[0], std::exp(-5.0), 1e-9);
}

// ── StepCallback ──────────────────────────────────────────────────────────────

TEST(ODE_Callback, RecordsTrajectory) {
    std::vector<real> times;
    std::vector<real> vals;

    ode_rk4(harmonic_osc(), {1.0, 0.0}, 0.0, 1.0, 0.1,
            [&](real t, const Vector& y) {
                times.push_back(t);
                vals.push_back(y[0]);
            });

    EXPECT_EQ(static_cast<int>(times.size()), 10);
    EXPECT_NEAR(times.back(), 1.0, 1e-12);
    EXPECT_NEAR(vals.back(), std::cos(1.0), 1e-6);  // h=0.1 RK4 gives ~1e-7 error
}

TEST(ODE_Callback, RK45CallbackCount) {
    int count = 0;
    ode_rk45(harmonic_osc(), {1.0, 0.0}, 0.0, 1.0, 1e-6, 1e-9, 1e-3, 100000,
             [&](real, const Vector&) { ++count; });
    EXPECT_GT(count, 0);
}

// ── Velocity Verlet ───────────────────────────────────────────────────────────

TEST(ODE_Verlet, CircularOrbit) {
    // Circular Kepler orbit at r=1 has period T = 2π
    auto res = ode_verlet(kepler_accel(), kepler_q0(), kepler_v0(),
                          0.0, 2.0 * M_PI, 1e-3);
    EXPECT_NEAR(res.q[0], 1.0, 1e-4);
    EXPECT_NEAR(res.q[1], 0.0, 1e-4);
    EXPECT_NEAR(res.v[0], 0.0, 1e-4);
    EXPECT_NEAR(res.v[1], 1.0, 1e-4);
}

TEST(ODE_Verlet, EnergyBounded) {
    // Verlet energy error stays bounded (oscillates, does not grow)
    real E0 = kepler_energy(kepler_q0(), kepler_v0());

    real max_drift = 0.0;
    ode_verlet(kepler_accel(), kepler_q0(), kepler_v0(), 0.0, 100.0, 0.01,
               [&](real, const Vector& q, const Vector& v) {
                   real E = kepler_energy(q, v);
                   max_drift = std::max(max_drift, std::abs(E - E0));
               });

    // Energy error must stay small — not growing secularly
    EXPECT_LT(max_drift, 1e-4);
}

TEST(ODE_Verlet, EnergyBetterThanRK4Long) {
    // Over a long integration, Verlet energy drift < RK4 energy drift
    real E0 = kepler_energy(kepler_q0(), kepler_v0());
    real h = 0.05;

    // RK4: pack y = [q; v]
    auto rhs = [](real, const Vector& y, Vector& dy) {
        real r2 = y[0]*y[0] + y[1]*y[1];
        real r3 = r2 * std::sqrt(r2);
        dy[0] =  y[2];  dy[1] =  y[3];
        dy[2] = -y[0]/r3; dy[3] = -y[1]/r3;
    };
    Vector y0{1.0, 0.0, 0.0, 1.0};
    real rk4_max_drift = 0.0;
    ode_rk4(rhs, y0, 0.0, 200.0, h,
            [&](real, const Vector& y) {
                Vector q{y[0],y[1]}, v{y[2],y[3]};
                rk4_max_drift = std::max(rk4_max_drift,
                                         std::abs(kepler_energy(q,v) - E0));
            });

    real verlet_max_drift = 0.0;
    ode_verlet(kepler_accel(), kepler_q0(), kepler_v0(), 0.0, 200.0, h,
               [&](real, const Vector& q, const Vector& v) {
                   verlet_max_drift = std::max(verlet_max_drift,
                                               std::abs(kepler_energy(q,v) - E0));
               });

    EXPECT_LT(verlet_max_drift, rk4_max_drift);
}

// ── Yoshida 4th order ─────────────────────────────────────────────────────────

TEST(ODE_Yoshida4, CircularOrbit) {
    auto res = ode_yoshida4(kepler_accel(), kepler_q0(), kepler_v0(),
                             0.0, 2.0 * M_PI, 0.05);
    EXPECT_NEAR(res.q[0], 1.0, 1e-4);
    EXPECT_NEAR(res.q[1], 0.0, 1e-4);
}

TEST(ODE_Yoshida4, HigherOrderThanVerlet) {
    // Yoshida4 should have smaller position error than Verlet for same h
    real h = 0.1;
    real T = 2.0 * M_PI;

    auto verlet_res  = ode_verlet  (kepler_accel(), kepler_q0(), kepler_v0(), 0.0, T, h);
    auto yoshida_res = ode_yoshida4(kepler_accel(), kepler_q0(), kepler_v0(), 0.0, T, h);

    real verlet_err  = std::hypot(verlet_res.q[0]  - 1.0, verlet_res.q[1]);
    real yoshida_err = std::hypot(yoshida_res.q[0] - 1.0, yoshida_res.q[1]);

    EXPECT_LT(yoshida_err, verlet_err);
}

TEST(ODE_Yoshida4, EnergyBounded) {
    real E0 = kepler_energy(kepler_q0(), kepler_v0());
    real max_drift = 0.0;
    ode_yoshida4(kepler_accel(), kepler_q0(), kepler_v0(), 0.0, 100.0, 0.05,
                 [&](real, const Vector& q, const Vector& v) {
                     max_drift = std::max(max_drift,
                                          std::abs(kepler_energy(q,v) - E0));
                 });
    EXPECT_LT(max_drift, 1e-8);
}

// ── SymplecticCallback ────────────────────────────────────────────────────────

TEST(ODE_Verlet, SymplecticCallbackFired) {
    int count = 0;
    ode_verlet(kepler_accel(), kepler_q0(), kepler_v0(), 0.0, 1.0, 0.1,
               [&](real, const Vector&, const Vector&) { ++count; });
    EXPECT_EQ(count, 10);
}
