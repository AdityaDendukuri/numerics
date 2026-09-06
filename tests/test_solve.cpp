/// @file tests/test_solve.cpp
/// @brief Coverage for the solve(problem, algorithm) and sample(model, sampler)
/// verbs: ODE returns an ode_result; MCMC samples a model carrying its observable.
#include "linear/sparse/sparse.hpp"
#include "linear/sparse/sparse_op.hpp"
#include "operator/properties.hpp"
#include "solve/sample.hpp"
#include "solve/solve.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <random>

using namespace num;

// solve(ode_problem, RK45{}) integrates y' = -y to y(1) = e^-1.
TEST(Solve, ODEProblemRK45) {
    ode_problem prob;
    prob.f = [](real, const vec &y, vec &dy) { dy[0] = -y[0]; };
    prob.u0 = vec{1.0};
    prob.t0 = 0.0;
    prob.tf = 1.0;

    const ode_result r = solve(prob, rk45_method{});
    EXPECT_TRUE(r.converged);
    EXPECT_NEAR(r.u[0], std::exp(-1.0), 1e-5);
}

// sample(mcmc_model, Metropolis, rng): the observable lives on the model, and the
// mean of a constant observable is that constant (deterministic).
TEST(Sample, MCMCModelMeasure) {
    std::mt19937 rng(42);

    mcmc_model model;
    model.n_sites = 10;
    model.accept_prob = [](int) { return 1.0; };
    model.propose = [](int) {};
    model.measure = []() { return 3.5; };

    const metropolis_method alg{.equilibration = 10, .measurements = 20};
    const double mean = sample(model, alg, rng);
    EXPECT_DOUBLE_EQ(mean, 3.5);
}

// init(problem, algorithm) + solve(cache) re-solves in place, warm-starting from
// the previous iterate; the result matches the one-shot solve(problem, algorithm).
TEST(Solve, LinearCacheWarmStart) {
    const auto A = spmat::from_triplets(3, 3, {0, 0, 1, 1, 1, 2, 2}, {0, 1, 0, 1, 2, 1, 2},
                                               {4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0});
    operators::sparse_op op(A);
    const auto Aspd = operators::assume_spd(op); // named: must outlive the cache
    const vec b{1.0, 2.0, 3.0};

    auto cache = init(linear_problem{Aspd, b}, cg_method{});
    const linear_solution s1 = solve(cache);
    ASSERT_TRUE(s1.converged);

    // Re-solving starts from the converged iterate: no more iterations, residual
    // stays tiny, solution unchanged. (CG's breakdown guard leaves .converged
    // unset when it starts already-solved, so assert on the residual instead.)
    const linear_solution s2 = solve(cache);
    EXPECT_LE(s2.iterations, s1.iterations);
    EXPECT_LT(s2.residual, 1e-6);
    EXPECT_NEAR(s2.u[0], s1.u[0], 1e-12);
    EXPECT_NEAR(s2.u[1], s1.u[1], 1e-12);
    EXPECT_NEAR(s2.u[2], s1.u[2], 1e-12);

    // Same solution as the one-shot verb.
    const linear_solution one = solve(linear_problem{Aspd, b}, cg_method{});
    EXPECT_NEAR(s1.u[0], one.u[0], 1e-9);
    EXPECT_NEAR(s1.u[1], one.u[1], 1e-9);
    EXPECT_NEAR(s1.u[2], one.u[2], 1e-9);
}

TEST(Solve, RestrictedPcgPreservesZeroSumProblemInvariant) {
    mat laplacian(2, 2, 0.0);
    laplacian(0, 0) = 1.0;
    laplacian(0, 1) = -1.0;
    laplacian(1, 0) = -1.0;
    laplacian(1, 1) = 1.0;
    mat identity(2, 2, 0.0);
    identity(0, 0) = 1.0;
    identity(1, 1) = 1.0;
    const auto restricted_A = num::assume<law::spd_on<space::zero_sum>>(laplacian);
    const auto restricted_M = num::assume<law::spd_on<space::zero_sum>>(identity);
    const vec b{1.0, -1.0};

    const auto solution =
        solve(linear_problem{restricted_A, b}, pcg_on_method{restricted_M, space::zero_sum{}});

    EXPECT_TRUE(solution.converged);
    EXPECT_TRUE(math::contains(space::zero_sum{}, solution.u));
    EXPECT_NEAR(solution.u[0], 0.5, 1e-12);
    EXPECT_NEAR(solution.u[1], -0.5, 1e-12);
}
