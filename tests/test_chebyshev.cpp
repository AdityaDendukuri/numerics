/// @file tests/test_chebyshev.cpp
/// @brief mat-free Chebyshev polynomial preconditioner.
///
/// The property that matters is positive definiteness: PCG's convergence theory
/// rests on it, and a Chebyshev polynomial loses it the moment the interval fails
/// to enclose the spectrum. The recurrence is also easy to write in a form that
/// looks plausible, converges, and is quietly indefinite — dropping the initial
/// direction term does exactly that — so the tests below check the polynomial's
/// definiteness directly rather than inferring it from a solve that happened to
/// finish.

#include "linear/solvers/chebyshev.hpp"
#include "container/matrix.hpp"
#include "linear/solvers/cg.hpp"
#include "linear/solvers/math_pcg.hpp"
#include "operator/callable.hpp"
#include "operator/dense.hpp"
#include "operator/properties.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <stdexcept>

namespace {

/// mat-free shifted 1D Laplacian: spectrum lies in [sigma, 4 + sigma].
auto shifted_laplacian(num::idx n, num::real sigma) {
    return num::operators::make_op(
        [sigma](const num::vec &u, num::vec &out) {
            const num::idx m = u.size();
            for (num::idx i = 0; i < m; ++i) {
                out[i] = ((2.0 + sigma) * u[i]) - (i > 0 ? u[i - 1] : 0.0) -
                         (i + 1 < m ? u[i + 1] : 0.0);
            }
        },
        n);
}

TEST(Chebyshev, EstimatesTheSpectralRadiusFromAbove) {
    const num::idx n = 500;
    const num::real sigma = 0.5;
    auto op = shifted_laplacian(n, sigma);
    // Largest eigenvalue of the shifted 1D Laplacian on n interior points.
    const num::real exact =
        sigma + (4.0 * std::pow(std::sin(M_PI * static_cast<double>(n) / (2.0 * (n + 1))), 2));

    const num::real estimate = num::estimate_largest_eigenvalue(op);
    EXPECT_GT(estimate, exact) << "an upper bound must not sit below the spectrum";
    EXPECT_LT(estimate, 1.5 * exact) << "and must not be uselessly loose";
}

TEST(Chebyshev, IsPositiveDefiniteWhenTheIntervalEnclosesTheSpectrum) {
    // assume_spd runs the sampled quadratic-form test; an indefinite polynomial
    // is rejected here rather than silently breaking the solve.
    const num::idx n = 300;
    const num::real sigma = 0.4;
    auto op = shifted_laplacian(n, sigma);
    auto A = num::operators::assume_spd(op);

    for (num::idx degree : {num::idx{1}, num::idx{2}, num::idx{4}, num::idx{8}}) {
        auto M = num::make_chebyshev_preconditioner(A, sigma, 4.0 + sigma, degree);
        EXPECT_NO_THROW((void)num::operators::assume_spd(M)) << "degree " << degree;
    }
}

TEST(Chebyshev, RejectsADegenerateOrNonPositiveInterval) {
    auto op = shifted_laplacian(50, 0.5);
    auto A = num::operators::assume_spd(op);
    EXPECT_THROW((void)num::make_chebyshev_preconditioner(A, 0.0, 4.0, 4), std::invalid_argument);
    EXPECT_THROW((void)num::make_chebyshev_preconditioner(A, -1.0, 4.0, 4), std::invalid_argument);
    EXPECT_THROW((void)num::make_chebyshev_preconditioner(A, 2.0, 2.0, 4), std::invalid_argument);
    EXPECT_THROW((void)num::make_chebyshev_preconditioner(A, 3.0, 1.0, 4), std::invalid_argument);
    EXPECT_THROW((void)num::make_chebyshev_preconditioner(A, 0.5, 4.0, 0), std::invalid_argument);
}

TEST(Chebyshev, DegreeOneIsTheScaledRichardsonStep) {
    const num::idx n = 64;
    auto op = shifted_laplacian(n, 1.0);
    auto A = num::operators::assume_spd(op);
    auto M = num::make_chebyshev_preconditioner(A, 1.0, 5.0, 1);

    num::vec r(n, 2.0);
    num::vec z(n, 0.0);
    M.apply(r, z);
    const num::real centre = 0.5 * (5.0 + 1.0);
    for (num::idx i = 0; i < n; ++i) {
        EXPECT_NEAR(z[i], 2.0 / centre, 1e-12) << "element " << i;
    }
}

TEST(Chebyshev, HigherDegreeApproximatesTheInverseMoreClosely) {
    // ||A p_m(A) r - r|| must fall as the degree rises: that is the whole claim.
    const num::idx n = 200;
    const num::real sigma = 0.5;
    auto op = shifted_laplacian(n, sigma);
    auto A = num::operators::assume_spd(op);

    num::vec r(n);
    for (num::idx i = 0; i < n; ++i) {
        r[i] = std::sin(0.3 * static_cast<double>(i));
    }

    num::real previous = std::numeric_limits<num::real>::max();
    for (num::idx degree : {num::idx{1}, num::idx{2}, num::idx{4}, num::idx{8}}) {
        auto M = num::make_chebyshev_preconditioner(A, sigma, 4.0 + sigma, degree);
        num::vec z(n, 0.0);
        num::vec back(n, 0.0);
        M.apply(r, z);
        num::math::apply(A, z, back);
        num::math::axpy(num::real(-1), r, back);
        const num::real defect = num::math::norm(back);
        EXPECT_LT(defect, previous) << "degree " << degree << " did not improve on the previous";
        previous = defect;
    }
}

TEST(Chebyshev, CutsKrylovIterationsRoughlyByTheDegree) {
    const num::idx n = 2000;
    const num::real sigma = 0.02; // condition number about 200
    auto op = shifted_laplacian(n, sigma);
    auto A = num::operators::assume_spd(op);

    num::vec b(n, 1.0);
    num::vec x_plain(n, 0.0);
    const auto plain =
        num::cg(A, b, x_plain, num::cg_options{.tolerance = 1e-10, .max_iterations = 20000});
    ASSERT_TRUE(plain.converged);

    auto M = num::make_chebyshev_preconditioner(A, sigma, 4.0 + sigma, 4);
    auto spd_precond = num::operators::assume_spd(M);
    num::vec x_prec(n, 0.0);
    const auto preconditioned = num::pcg(
        A, spd_precond, b, x_prec, num::pcg_options{.tolerance = 1e-10, .max_iterations = 20000});
    ASSERT_TRUE(preconditioned.converged);

    // A degree-m polynomial buys about a factor of m; allow generous slack.
    EXPECT_LT(preconditioned.iterations * 3, plain.iterations)
        << "expected roughly a 4x reduction, got " << plain.iterations << " -> "
        << preconditioned.iterations;

    // Both must reach the same solution.
    for (num::idx i = 0; i < n; i += 97) {
        EXPECT_NEAR(x_prec[i], x_plain[i], 1e-6 * std::max(1.0, std::abs(x_plain[i])));
    }
}

TEST(Chebyshev, WorksOnAnExplicitDenseOperatorToo) {
    // Nothing about the preconditioner is matrix-free-only; it needs only apply().
    const num::idx n = 40;
    num::mat A(n, n, 0.0);
    for (num::idx i = 0; i < n; ++i) {
        A(i, i) = 4.0;
        if (i > 0) {
            A(i, i - 1) = -1.0;
        }
        if (i + 1 < n) {
            A(i, i + 1) = -1.0;
        }
    }
    num::operators::dense_op op(A);
    auto spd = num::operators::assume_spd(op);
    auto M = num::make_chebyshev_preconditioner(spd, 2.0, 6.0, 4);
    EXPECT_NO_THROW((void)num::operators::assume_spd(M));
    EXPECT_EQ(M.rows(), n);
    EXPECT_EQ(M.degree(), 4u);
}

TEST(Chebyshev, RepeatedApplicationIsStateless) {
    // The workspace is a mutable member reused across calls; a leftover residual
    // would make the second application differ from the first.
    const num::idx n = 128;
    auto op = shifted_laplacian(n, 0.5);
    auto A = num::operators::assume_spd(op);
    auto M = num::make_chebyshev_preconditioner(A, 0.5, 4.5, 6);

    num::vec r(n, 1.5);
    num::vec first(n, 0.0);
    num::vec second(n, 0.0);
    M.apply(r, first);
    M.apply(r, second);
    for (num::idx i = 0; i < n; ++i) {
        EXPECT_DOUBLE_EQ(first[i], second[i]) << "element " << i;
    }
}

} // namespace
