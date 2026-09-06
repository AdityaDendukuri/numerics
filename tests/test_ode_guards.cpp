/// @file tests/test_ode_guards.cpp
/// @brief Integration parameters that used to hang or silently lie.
///
/// Every stepper advances by `min(h, t1 - t)` and stops when `t` reaches `t1`.
/// Two ordinary-looking parameter values broke that without saying so: `h = 0`
/// advanced time by nothing and spun forever, and `tf < t0` satisfied the stop
/// condition immediately, returning the initial condition with `converged` set
/// to true. The fixed-step integrators also ignored `max_steps` entirely, so
/// there was no work limit to fall back on.
///
/// Each case below is an input a user could plausibly supply.

#include "ode/ode.hpp"
#include "ode/steps.hpp"
#include <gtest/gtest.h>
#include <stdexcept>

namespace {

/// Decay problem with a known closed form: y' = -y, y(0) = 1.
void decay(num::real, const num::vec &y, num::vec &dy) {
    dy[0] = -y[0];
}

/// Harmonic acceleration for the second-order integrators: q'' = -q.
void spring(const num::vec &q, num::vec &a) {
    a[0] = -q[0];
}

num::ode_params good() {
    return num::ode_params{.t0 = 0.0, .tf = 1.0, .h = 1e-2, .rtol = 1e-8, .atol = 1e-10};
}

// --- a zero step used to spin forever ---------------------------------------

TEST(ODEGuards, ZeroStepIsRejectedByEveryIntegrator) {
    auto p = good();
    p.h = 0.0;
    const num::vec y0{1.0};
    const num::vec v0{0.0};

    EXPECT_THROW(num::ode_euler(decay, y0, p), std::invalid_argument);
    EXPECT_THROW(num::ode_rk4(decay, y0, p), std::invalid_argument);
    EXPECT_THROW(num::ode_rk45(decay, y0, p), std::invalid_argument);
    EXPECT_THROW(num::ode_verlet(spring, y0, v0, p), std::invalid_argument);
    EXPECT_THROW(num::ode_yoshida4(spring, y0, v0, p), std::invalid_argument);
}

TEST(ODEGuards, NegativeAndNonFiniteStepAreRejected) {
    const num::vec y0{1.0};
    auto negative = good();
    negative.h = -1e-2;
    EXPECT_THROW(num::ode_rk4(decay, y0, negative), std::invalid_argument);

    auto infinite = good();
    infinite.h = std::numeric_limits<num::real>::infinity();
    EXPECT_THROW(num::ode_rk4(decay, y0, infinite), std::invalid_argument);
}

// --- backward integration used to succeed while doing nothing ---------------

TEST(ODEGuards, BackwardIntervalIsRejectedRatherThanSilentlySkipped) {
    auto p = good();
    p.t0 = 1.0;
    p.tf = 0.0;
    const num::vec y0{1.0};
    const num::vec v0{0.0};

    EXPECT_THROW(num::ode_euler(decay, y0, p), std::invalid_argument);
    EXPECT_THROW(num::ode_rk4(decay, y0, p), std::invalid_argument);
    EXPECT_THROW(num::ode_rk45(decay, y0, p), std::invalid_argument);
    EXPECT_THROW(num::ode_verlet(spring, y0, v0, p), std::invalid_argument);
}

TEST(ODEGuards, NonFiniteEndpointsAreRejected) {
    auto p = good();
    p.tf = std::numeric_limits<num::real>::quiet_NaN();
    EXPECT_THROW(num::ode_rk4(decay, num::vec{1.0}, p), std::invalid_argument);
}

TEST(ODEGuards, ZeroWorkLimitIsRejected) {
    auto p = good();
    p.max_steps = 0;
    EXPECT_THROW(num::ode_rk4(decay, num::vec{1.0}, p), std::invalid_argument);
}

// --- fixed-step integrators used to always claim success --------------------

TEST(ODEGuards, ExhaustingTheWorkLimitReportsFailureRatherThanSuccess) {
    auto p = good();
    p.h = 1e-4;      // 10,000 steps are needed
    p.max_steps = 5; // only 5 are allowed
    const auto result = num::ode_rk4(decay, num::vec{1.0}, p);

    EXPECT_FALSE(result.converged) << "stopping early must not be reported as success";
    EXPECT_EQ(result.steps, 5u);
    EXPECT_LT(result.t, p.tf) << "the reported time must be where it stopped, not where it aimed";
    EXPECT_NEAR(result.t, 5e-4, 1e-12);
}

TEST(ODEGuards, ACompletedIntegrationStillReportsSuccessAndTheRightAnswer) {
    const auto result = num::ode_rk4(decay, num::vec{1.0}, good());
    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.t, 1.0, 1e-12);
    EXPECT_NEAR(result.u[0], std::exp(-1.0), 1e-8) << "y(1) = e^-1";
}

TEST(ODEGuards, EulerReportsTheTimeItActuallyReached) {
    auto p = good();
    p.h = 1e-3;
    p.max_steps = 100;
    const auto result = num::ode_euler(decay, num::vec{1.0}, p);
    EXPECT_FALSE(result.converged);
    EXPECT_NEAR(result.t, 0.1, 1e-12);
}

TEST(ODEGuards, AdaptiveIntegratorStillConvergesOnAnOrdinaryProblem) {
    const auto result = num::ode_rk45(decay, num::vec{1.0}, good());
    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.u[0], std::exp(-1.0), 1e-7);
}

} // namespace
