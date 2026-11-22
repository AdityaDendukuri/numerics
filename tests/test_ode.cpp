/// @file test_ode.cpp
/// @brief Tests for ODE integrators: convergence, energy conservation, stepper
/// ranges.
#include "ode/ode.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <vector>

using namespace num;

// Helpers

static ode_rhs_fn harmonic_osc() {
    return [](real t, const vec<real> &y, vec<real> &dy) {
        (void)t;
        dy[0] = y[1];
        dy[1] = -y[0];
    };
}

static accel_fn kepler_accel() {
    return [](const vec<real> &q, vec<real> &a) {
        real r2 = (q[0] * q[0]) + (q[1] * q[1]);
        real r3 = r2 * std::sqrt(r2);
        a[0] = -q[0] / r3;
        a[1] = -q[1] / r3;
    };
}

static real kepler_energy(const vec<real> &q, const vec<real> &v) {
    real KE = 0.5 * ((v[0] * v[0]) + (v[1] * v[1]));
    real r = std::sqrt((q[0] * q[0]) + (q[1] * q[1]));
    return KE - (1.0 / r);
}

static vec<real> kepler_q0() {
    return {1.0, 0.0};
}
static vec<real> kepler_v0() {
    return {0.0, 1.0};
}

// Euler

TEST(ODE_Euler, OrderOne) {
    auto f = [](real t, const vec<real> &y, vec<real> &dy) {
        (void)t;
        dy[0] = -y[0];
    };

    auto err = [&](real h) {
        auto res = ode_euler(f, {1.0}, {.tf = 1.0, .h = h});
        return std::abs(res.u[0] - std::exp(-1.0));
    };

    real e1 = err(0.01), e2 = err(0.005);
    EXPECT_NEAR(std::log(e1 / e2) / std::log(2.0), 1.0, 0.05);
}

// RK4

TEST(ODE_RK4, OrderFour) {
    auto f = harmonic_osc();
    vec<real> y0{1.0, 0.0};

    auto err = [&](real h) {
        auto res = ode_rk4(f, y0, {.tf = 1.0, .h = h});
        return std::abs(res.u[0] - std::cos(1.0));
    };

    real e1 = err(0.1), e2 = err(0.05);
    EXPECT_NEAR(std::log(e1 / e2) / std::log(2.0), 4.0, 0.1);
}

TEST(ODE_RK4, HarmonicOscillatorAccuracy) {
    auto res = ode_rk4(harmonic_osc(), {1.0, 0.0}, {.tf = 2.0 * num::pi, .h = 0.01});
    EXPECT_NEAR(res.u[0], 1.0, 1e-8);
    EXPECT_NEAR(res.u[1], 0.0, 1e-8);
}

// RK45

TEST(ODE_RK45, HarmonicOscillator) {
    auto res =
        ode_rk45(harmonic_osc(), {1.0, 0.0}, {.tf = 2.0 * num::pi, .rtol = 1e-9, .atol = 1e-12});
    EXPECT_TRUE(res.converged);
    EXPECT_NEAR(res.u[0], 1.0, 1e-7);
    EXPECT_NEAR(res.u[1], 0.0, 1e-7);
}

TEST(ODE_RK45, ExponentialDecay) {
    auto f = [](real t, const vec<real> &y, vec<real> &dy) {
        (void)t;
        dy[0] = -y[0];
    };
    auto res = ode_rk45(f, {1.0}, {.tf = 5.0, .rtol = 1e-10, .atol = 1e-12});
    EXPECT_NEAR(res.u[0], std::exp(-5.0), 1e-9);
}

// Stepper ranges

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
    for (auto s : rk45(harmonic_osc(), {1.0, 0.0},
                       {.tf = 1.0, .rtol = 1e-6, .atol = 1e-9, .max_steps = 100000})) {
        (void)s, ++count;
    }
    EXPECT_GT(count, 0);
}

// Velocity Verlet

TEST(ODE_Verlet, CircularOrbit) {
    auto res = ode_verlet(kepler_accel(), kepler_q0(), kepler_v0(), {.tf = 2.0 * num::pi, .h = 1e-3});
    EXPECT_NEAR(res.q[0], 1.0, 1e-4);
    EXPECT_NEAR(res.q[1], 0.0, 1e-4);
    EXPECT_NEAR(res.v[0], 0.0, 1e-4);
    EXPECT_NEAR(res.v[1], 1.0, 1e-4);
}

TEST(ODE_Verlet, EnergyBounded) {
    real E0 = kepler_energy(kepler_q0(), kepler_v0());
    real max_drift = 0.0;

    for (auto [t, q, v] :
         verlet(kepler_accel(), kepler_q0(), kepler_v0(), {.tf = 100.0, .h = 0.01})) {
        (void)t;
        max_drift = std::max(max_drift, std::abs(kepler_energy(q, v) - E0));
    }

    EXPECT_LT(max_drift, 1e-4);
}

TEST(ODE_Verlet, EnergyBetterThanRK4Long) {
    real E0 = kepler_energy(kepler_q0(), kepler_v0());
    real h = 0.05;

    auto rhs = [](real, const vec<real> &y, vec<real> &dy) {
        real r2 = (y[0] * y[0]) + (y[1] * y[1]);
        real r3 = r2 * std::sqrt(r2);
        dy[0] = y[2];
        dy[1] = y[3];
        dy[2] = -y[0] / r3;
        dy[3] = -y[1] / r3;
    };
    real rk4_max_drift = 0.0;
    for (auto [t, y] : rk4(rhs, vec<real>{1.0, 0.0, 0.0, 1.0}, {.tf = 200.0, .h = h})) {
        (void)t;
        vec<real> q{y[0], y[1]}, v{y[2], y[3]};
        rk4_max_drift = std::max(rk4_max_drift, std::abs(kepler_energy(q, v) - E0));
    }

    real verlet_max_drift = 0.0;
    for (auto [t, q, v] : verlet(kepler_accel(), kepler_q0(), kepler_v0(), {.tf = 200.0, .h = h})) {
        (void)t;
        verlet_max_drift = std::max(verlet_max_drift, std::abs(kepler_energy(q, v) - E0));
    }

    EXPECT_LT(verlet_max_drift, rk4_max_drift);
}

// Yoshida 4th order

TEST(ODE_Yoshida4, CircularOrbit) {
    auto res =
        ode_yoshida4(kepler_accel(), kepler_q0(), kepler_v0(), {.tf = 2.0 * num::pi, .h = 0.05});
    EXPECT_NEAR(res.q[0], 1.0, 1e-4);
    EXPECT_NEAR(res.q[1], 0.0, 1e-4);
}

TEST(ODE_Yoshida4, HigherOrderThanVerlet) {
    real T = 2.0 * num::pi;
    auto verlet_res = ode_verlet(kepler_accel(), kepler_q0(), kepler_v0(), {.tf = T, .h = 0.1});
    auto yoshida_res = ode_yoshida4(kepler_accel(), kepler_q0(), kepler_v0(), {.tf = T, .h = 0.1});

    EXPECT_LT(std::hypot(yoshida_res.q[0] - 1.0, yoshida_res.q[1]),
              std::hypot(verlet_res.q[0] - 1.0, verlet_res.q[1]));
}

TEST(ODE_Yoshida4, EnergyBounded) {
    real E0 = kepler_energy(kepler_q0(), kepler_v0());
    real max_drift = 0.0;

    for (auto [t, q, v] :
         yoshida4(kepler_accel(), kepler_q0(), kepler_v0(), {.tf = 100.0, .h = 0.05})) {
        (void)t;
        max_drift = std::max(max_drift, std::abs(kepler_energy(q, v) - E0));
    }

    EXPECT_LT(max_drift, 1e-8);
}

TEST(ODE_Verlet, StepCount) {
    int count = 0;
    for (auto s : verlet(kepler_accel(), kepler_q0(), kepler_v0(), {.tf = 1.0, .h = 0.1})) {
        (void)s, ++count;
    }
    EXPECT_EQ(count, 10);
}

// Stiffness: classical RK4 is stable on y' = lambda y only for h|lambda| below 2.785, whatever
// accuracy the smooth solution would allow. One step must multiply by exactly R(h lambda).
TEST(ODE_RK4, StabilityBoundaryOnTheNegativeRealAxis) {
    const real lambda = -1000.0;
    auto f = [lambda](real, const vec<real> &y, vec<real> &dy) { dy[0] = lambda * y[0]; };
    auto R = [](real z) { return 1 + z + z * z / 2 + z * z * z / 6 + z * z * z * z / 24; };
    for (real hz : {2.7, 2.9}) {
        const real h = hz / -lambda;
        const auto one = ode_rk4(f, vec<real>{1.0}, {.t0 = 0.0, .tf = h, .h = h});
        EXPECT_NEAR(one.u[0], R(-hz), 1e-14);
        const auto many = ode_rk4(f, vec<real>{1.0}, {.t0 = 0.0, .tf = 200 * h, .h = h});
        if (hz < 2.785) {
            EXPECT_LT(std::abs(many.u[0]), 1e-10);
        } else {
            EXPECT_GT(std::abs(many.u[0]), 1e10);
        }
    }
}

// t += h with h = 0.1 accumulates to 0.30000000000000004 or 0.9999999999999999; the driver
// must neither take an extra sliver step nor stop one step short.
TEST(ODE_RK4, DoesNotTakeASliverStep) {
    auto f = [](real, const vec<real> &y, vec<real> &dy) { dy[0] = y[0]; };
    for (const auto &[tf, steps] : {std::pair{0.3, idx{3}}, std::pair{1.0, idx{10}}, std::pair{0.7, idx{7}}}) {
        const auto r = ode_rk4(f, vec<real>{1.0}, {.t0 = 0.0, .tf = tf, .h = 0.1});
        EXPECT_EQ(r.steps, steps) << "tf = " << tf;
        EXPECT_NEAR(r.t, tf, 1e-14) << "tf = " << tf;
        EXPECT_NEAR(r.u[0], std::exp(tf), 1e-5) << "tf = " << tf;
    }
}

// Prothero-Robinson y' = lambda (y - cos t) - sin t with lambda = -1e4: the solution cos t is
// smooth, but an explicit adaptive method is stability-limited and must still deliver it.
TEST(ODE_RK45, SurvivesProtheroRobinsonStiffness) {
    const real lambda = -1e4;
    auto f = [lambda](real t, const vec<real> &y, vec<real> &dy) {
        dy[0] = lambda * (y[0] - std::cos(t)) - std::sin(t);
    };
    const auto r = ode_rk45(f, vec<real>{1.0}, {.t0 = 0.0, .tf = 1.0, .h = 1e-3, .rtol = 1e-8, .atol = 1e-10});
    EXPECT_TRUE(r.converged) << r;
    EXPECT_NEAR(r.u[0], std::cos(1.0), 1e-6);
}

// The adaptive controller's error norm is a running max; dropping a NaN would accept the step
// and report success with a NaN state.
TEST(ODE_RK45, ReportsFailureOnANonFiniteState) {
    auto f = [](real t, const vec<real> &y, vec<real> &dy) {
        dy[0] = -y[0];
        dy[1] = t > 0.5 ? num::nan : 1.0;
    };
    const auto r = ode_rk45(f, vec<real>{1.0, 0.0}, {.t0 = 0.0, .tf = 1.0, .h = 1e-2});
    EXPECT_FALSE(r.converged) << r;
    EXPECT_LT(r.t, 1.0);
}
