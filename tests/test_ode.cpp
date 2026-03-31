/// @file test_ode.cpp
/// @brief Tests for ODE integrators: convergence, energy conservation, stepper
/// ranges.
#include <gtest/gtest.h>
#include "ode/ode.hpp"
#include <cmath>
#include <vector>

using namespace num;

// ── Helpers
// ───────────────────────────────────────────────────────────────────

static ODERhsFn harmonic_osc() {
    return [](real t, const Vector& y, Vector& dy) {
        (void)t;
        dy[0] = y[1];
        dy[1] = -y[0];
    };
}

static AccelFn kepler_accel() {
    return [](const Vector& q, Vector& a) {
        real r2 = q[0] * q[0] + q[1] * q[1];
        real r3 = r2 * std::sqrt(r2);
        a[0]    = -q[0] / r3;
        a[1]    = -q[1] / r3;
    };
}

static real kepler_energy(const Vector& q, const Vector& v) {
    real KE = 0.5 * (v[0] * v[0] + v[1] * v[1]);
    real r  = std::sqrt(q[0] * q[0] + q[1] * q[1]);
    return KE - 1.0 / r;
}

static Vector kepler_q0() {
    return {1.0, 0.0};
}
static Vector kepler_v0() {
    return {0.0, 1.0};
}

// ── Euler
// ─────────────────────────────────────────────────────────────────────

TEST(ODE_Euler, OrderOne) {
    auto f = [](real t, const Vector& y, Vector& dy) {
        (void)t;
        dy[0] = -y[0];
    };

    auto err = [&](real h) {
        auto res = ode_euler(f, {1.0}, {.tf = 1.0, .h = h});
        return std::abs(res.y[0] - std::exp(-1.0));
    };

    real e1 = err(0.01), e2 = err(0.005);
    EXPECT_NEAR(std::log(e1 / e2) / std::log(2.0), 1.0, 0.05);
}

// ── RK4
// ───────────────────────────────────────────────────────────────────────

TEST(ODE_RK4, OrderFour) {
    auto   f = harmonic_osc();
    Vector y0{1.0, 0.0};

    auto err = [&](real h) {
        auto res = ode_rk4(f, y0, {.tf = 1.0, .h = h});
        return std::abs(res.y[0] - std::cos(1.0));
    };

    real e1 = err(0.1), e2 = err(0.05);
    EXPECT_NEAR(std::log(e1 / e2) / std::log(2.0), 4.0, 0.1);
}

TEST(ODE_RK4, HarmonicOscillatorAccuracy) {
    auto res =
        ode_rk4(harmonic_osc(), {1.0, 0.0}, {.tf = 2.0 * M_PI, .h = 0.01});
    EXPECT_NEAR(res.y[0], 1.0, 1e-8);
    EXPECT_NEAR(res.y[1], 0.0, 1e-8);
}

// ── RK45
// ──────────────────────────────────────────────────────────────────────

TEST(ODE_RK45, HarmonicOscillator) {
    auto res = ode_rk45(harmonic_osc(),
                        {1.0, 0.0},
                        {.tf = 2.0 * M_PI, .rtol = 1e-9, .atol = 1e-12});
    EXPECT_TRUE(res.converged);
    EXPECT_NEAR(res.y[0], 1.0, 1e-7);
    EXPECT_NEAR(res.y[1], 0.0, 1e-7);
}

TEST(ODE_RK45, ExponentialDecay) {
    auto f = [](real t, const Vector& y, Vector& dy) {
        (void)t;
        dy[0] = -y[0];
    };
    auto res = ode_rk45(f, {1.0}, {.tf = 5.0, .rtol = 1e-10, .atol = 1e-12});
    EXPECT_NEAR(res.y[0], std::exp(-5.0), 1e-9);
}

// ── Stepper ranges
// ────────────────────────────────────────────────────────────

TEST(ODE_Stepper, RK4RecordsTrajectory) {
    std::vector<real> times;
    std::vector<real> vals;

    for (auto [t, y] : rk4(harmonic_osc(), {1.0, 0.0}, {.tf = 1.0, .h = 0.1})) {
        times.push_back(t);
        vals.push_back(y[0]);
    }

    EXPECT_EQ(static_cast<int>(times.size()), 10);
    EXPECT_NEAR(times.back(), 1.0, 1e-12);
    EXPECT_NEAR(vals.back(), std::cos(1.0), 1e-6);
}

TEST(ODE_Stepper, RK45StepCount) {
    int count = 0;
    for (auto s :
         rk45(harmonic_osc(),
              {1.0, 0.0},
              {.tf = 1.0, .rtol = 1e-6, .atol = 1e-9, .max_steps = 100000}))
        (void)s, ++count;
    EXPECT_GT(count, 0);
}

// ── Velocity Verlet
// ───────────────────────────────────────────────────────────

TEST(ODE_Verlet, CircularOrbit) {
    auto res = ode_verlet(kepler_accel(),
                          kepler_q0(),
                          kepler_v0(),
                          {.tf = 2.0 * M_PI, .h = 1e-3});
    EXPECT_NEAR(res.q[0], 1.0, 1e-4);
    EXPECT_NEAR(res.q[1], 0.0, 1e-4);
    EXPECT_NEAR(res.v[0], 0.0, 1e-4);
    EXPECT_NEAR(res.v[1], 1.0, 1e-4);
}

TEST(ODE_Verlet, EnergyBounded) {
    real E0        = kepler_energy(kepler_q0(), kepler_v0());
    real max_drift = 0.0;

    for (auto [t, q, v] : verlet(kepler_accel(),
                                 kepler_q0(),
                                 kepler_v0(),
                                 {.tf = 100.0, .h = 0.01})) {
        (void)t;
        max_drift = std::max(max_drift, std::abs(kepler_energy(q, v) - E0));
    }

    EXPECT_LT(max_drift, 1e-4);
}

TEST(ODE_Verlet, EnergyBetterThanRK4Long) {
    real E0 = kepler_energy(kepler_q0(), kepler_v0());
    real h  = 0.05;

    auto rhs = [](real, const Vector& y, Vector& dy) {
        real r2 = y[0] * y[0] + y[1] * y[1];
        real r3 = r2 * std::sqrt(r2);
        dy[0]   = y[2];
        dy[1]   = y[3];
        dy[2]   = -y[0] / r3;
        dy[3]   = -y[1] / r3;
    };
    real rk4_max_drift = 0.0;
    for (auto [t, y] :
         rk4(rhs, Vector{1.0, 0.0, 0.0, 1.0}, {.tf = 200.0, .h = h})) {
        (void)t;
        Vector q{y[0], y[1]}, v{y[2], y[3]};
        rk4_max_drift = std::max(rk4_max_drift,
                                 std::abs(kepler_energy(q, v) - E0));
    }

    real verlet_max_drift = 0.0;
    for (auto [t, q, v] : verlet(kepler_accel(),
                                 kepler_q0(),
                                 kepler_v0(),
                                 {.tf = 200.0, .h = h})) {
        (void)t;
        verlet_max_drift = std::max(verlet_max_drift,
                                    std::abs(kepler_energy(q, v) - E0));
    }

    EXPECT_LT(verlet_max_drift, rk4_max_drift);
}

// ── Yoshida 4th order
// ─────────────────────────────────────────────────────────

TEST(ODE_Yoshida4, CircularOrbit) {
    auto res = ode_yoshida4(kepler_accel(),
                            kepler_q0(),
                            kepler_v0(),
                            {.tf = 2.0 * M_PI, .h = 0.05});
    EXPECT_NEAR(res.q[0], 1.0, 1e-4);
    EXPECT_NEAR(res.q[1], 0.0, 1e-4);
}

TEST(ODE_Yoshida4, HigherOrderThanVerlet) {
    real T           = 2.0 * M_PI;
    auto verlet_res  = ode_verlet(kepler_accel(),
                                 kepler_q0(),
                                 kepler_v0(),
                                  {.tf = T, .h = 0.1});
    auto yoshida_res = ode_yoshida4(kepler_accel(),
                                    kepler_q0(),
                                    kepler_v0(),
                                    {.tf = T, .h = 0.1});

    EXPECT_LT(std::hypot(yoshida_res.q[0] - 1.0, yoshida_res.q[1]),
              std::hypot(verlet_res.q[0] - 1.0, verlet_res.q[1]));
}

TEST(ODE_Yoshida4, EnergyBounded) {
    real E0        = kepler_energy(kepler_q0(), kepler_v0());
    real max_drift = 0.0;

    for (auto [t, q, v] : yoshida4(kepler_accel(),
                                   kepler_q0(),
                                   kepler_v0(),
                                   {.tf = 100.0, .h = 0.05})) {
        (void)t;
        max_drift = std::max(max_drift, std::abs(kepler_energy(q, v) - E0));
    }

    EXPECT_LT(max_drift, 1e-8);
}

TEST(ODE_Verlet, StepCount) {
    int count = 0;
    for (
        auto s :
        verlet(kepler_accel(), kepler_q0(), kepler_v0(), {.tf = 1.0, .h = 0.1}))
        (void)s, ++count;
    EXPECT_EQ(count, 10);
}
