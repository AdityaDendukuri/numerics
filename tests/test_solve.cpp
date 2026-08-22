/// @file tests/test_solve.cpp
/// @brief Coverage for the solve(problem, algorithm) and sample(model, sampler)
/// verbs: ODE returns an ODEResult; MCMC samples a model carrying its observable.
#include "linalg/sparse/sparse.hpp"
#include "linalg/sparse/sparse_op.hpp"
#include "operator/properties.hpp"
#include "solve/sample.hpp"
#include "solve/solve.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <random>

using namespace num;

// solve(ODEProblem, RK45{}) integrates y' = -y to y(1) = e^-1.
TEST(Solve, ODEProblemRK45) {
    ODEProblem prob;
    prob.f = [](real, const Vector &y, Vector &dy) { dy[0] = -y[0]; };
    prob.u0 = Vector{1.0};
    prob.t0 = 0.0;
    prob.tf = 1.0;

    const ODEResult r = solve(prob, RK45{});
    EXPECT_TRUE(r.converged);
    EXPECT_NEAR(r.u[0], std::exp(-1.0), 1e-5);
}

// sample(MCMCModel, Metropolis, rng): the observable lives on the model, and the
// mean of a constant observable is that constant (deterministic).
TEST(Sample, MCMCModelMeasure) {
    std::mt19937 rng(42);

    MCMCModel model;
    model.n_sites = 10;
    model.accept_prob = [](int) { return 1.0; };
    model.propose = [](int) {};
    model.measure = []() { return 3.5; };

    const Metropolis alg{.equilibration = 10, .measurements = 20};
    const double mean = sample(model, alg, rng);
    EXPECT_DOUBLE_EQ(mean, 3.5);
}

// init(problem, algorithm) + solve(cache) re-solves in place, warm-starting from
// the previous iterate; the result matches the one-shot solve(problem, algorithm).
TEST(Solve, LinearCacheWarmStart) {
    const auto A = SparseMatrix::from_triplets(3, 3, {0, 0, 1, 1, 1, 2, 2}, {0, 1, 0, 1, 2, 1, 2},
                                               {4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0});
    operators::SparseOp op(A);
    const auto Aspd = operators::assume_spd(op); // named: must outlive the cache
    const Vector b{1.0, 2.0, 3.0};

    auto cache = init(LinearProblem{Aspd, b}, CG{});
    const LinearSolution s1 = solve(cache);
    ASSERT_TRUE(s1.converged);

    // Re-solving starts from the converged iterate: no more iterations, residual
    // stays tiny, solution unchanged. (CG's breakdown guard leaves .converged
    // unset when it starts already-solved, so assert on the residual instead.)
    const LinearSolution s2 = solve(cache);
    EXPECT_LE(s2.iterations, s1.iterations);
    EXPECT_LT(s2.residual, 1e-6);
    EXPECT_NEAR(s2.u[0], s1.u[0], 1e-12);
    EXPECT_NEAR(s2.u[1], s1.u[1], 1e-12);
    EXPECT_NEAR(s2.u[2], s1.u[2], 1e-12);

    // Same solution as the one-shot verb.
    const LinearSolution one = solve(LinearProblem{Aspd, b}, CG{});
    EXPECT_NEAR(s1.u[0], one.u[0], 1e-9);
    EXPECT_NEAR(s1.u[1], one.u[1], 1e-9);
    EXPECT_NEAR(s1.u[2], one.u[2], 1e-9);
}
