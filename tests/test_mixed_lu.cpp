/// @file tests/test_mixed_lu.cpp
/// @brief Single-precision LU refined to double-precision accuracy.

#include "linear/condition.hpp"
#include "linear/factorization/factor.hpp"
#include "linear/factorization/mixed_lu.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <random>

using namespace num;

namespace {

mat<real> random_matrix(idx n, unsigned seed, real diagonal_boost) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<real> entry(-1.0, 1.0);
    mat<real> A(n, n, 0.0);
    for (idx i = 0; i < n; ++i) {
        for (idx j = 0; j < n; ++j) {
            A(i, j) = entry(generator);
        }
        A(i, i) += diagonal_boost;
    }
    return A;
}

// ||b - op(A) x||_inf / (||A||_inf ||x||_inf): what a backward-stable double solve achieves.
real backward_error(const mat<real> &A, bool transposed, const vec<real> &b, const vec<real> &x) {
    const idx n = A.rows();
    real residual = 0.0, norm_A = 0.0, norm_x = 0.0;
    for (idx i = 0; i < n; ++i) {
        real Ax = 0.0, row = 0.0;
        for (idx j = 0; j < n; ++j) {
            const real a = transposed ? A(j, i) : A(i, j);
            Ax += a * x[j];
            row += std::abs(a);
        }
        residual = std::max(residual, std::abs(b[i] - Ax));
        norm_A = std::max(norm_A, row);
        norm_x = std::max(norm_x, std::abs(x[i]));
    }
    return residual / (norm_A * norm_x);
}

vec<real> ramp(idx n) {
    vec<real> b(n);
    for (idx i = 0; i < n; ++i) {
        b[i] = std::sin(static_cast<real>(i + 1));
    }
    return b;
}

} // namespace

static_assert(factorization<mixed_lu>);

// A float factorization alone leaves a backward error near 1e-7; refinement reaches double's.
TEST(MixedLU, RefinesToDoublePrecision) {
    const idx n = 80;
    const mat<real> A = random_matrix(n, 3, 4.0);
    const vec<real> b = ramp(n);
    const mixed_lu F = lu(A, mixed_precision);
    ASSERT_TRUE(F.refined());

    vec<float> x_float;
    solve(F.low, detail::converted<float>(b), x_float);
    EXPECT_GT(backward_error(A, false, b, detail::converted<real>(x_float)), 1e-10);

    vec<real> x;
    solve(F, b, x);
    EXPECT_LT(backward_error(A, false, b, x), 10 * std::numeric_limits<real>::epsilon());

    vec<real> xt;
    solve_transpose(F, b, xt);
    EXPECT_LT(backward_error(A, true, b, xt), 10 * std::numeric_limits<real>::epsilon());
}

TEST(MixedLU, MatchesDoubleLU) {
    const idx n = 50;
    const mat<real> A = random_matrix(n, 11, 3.0);
    const vec<real> b = ramp(n);
    vec<real> x, expected;
    solve(lu(A, mixed_precision), b, x);
    solve(lu(A), b, expected);
    for (idx i = 0; i < n; ++i) {
        EXPECT_NEAR(x[i], expected[i], 1e-12 * (1.0 + std::abs(expected[i])));
    }

    mat<real> B(n, 3, 0.0), X;
    for (idx i = 0; i < n; ++i) {
        for (idx c = 0; c < 3; ++c) {
            B(i, c) = b[i] * static_cast<real>(c + 1);
        }
    }
    solve(lu(A, mixed_precision), B, X);
    for (idx i = 0; i < n; ++i) {
        for (idx c = 0; c < 3; ++c) {
            EXPECT_NEAR(X(i, c), expected[i] * static_cast<real>(c + 1),
                        1e-12 * (1.0 + std::abs(expected[i])));
        }
    }
}

// Too ill-conditioned for float: refinement would not converge, so the factor uses double.
TEST(MixedLU, FallsBackToDoubleWhenIllConditioned) {
    const idx n = 10;
    mat<real> H(n, n, 0.0);
    for (idx i = 0; i < n; ++i) {
        for (idx j = 0; j < n; ++j) {
            H(i, j) = 1.0 / static_cast<real>(i + j + 1);
        }
    }
    const mixed_lu F = lu(H, mixed_precision);
    EXPECT_FALSE(F.refined());
    EXPECT_GT(F.condition_estimate, 1e7);

    const vec<real> b = ramp(n);
    vec<real> x;
    solve(F, b, x);
    EXPECT_LT(backward_error(H, false, b, x), 100 * std::numeric_limits<real>::epsilon());
}

TEST(MixedLU, WorksWithGenericFactorizationTools) {
    const idx n = 30;
    const mat<real> A = random_matrix(n, 5, 4.0);
    const mixed_lu F = lu(A, mixed_precision);
    const vec<real> b = ramp(n);
    const vec<real> via_view = solve(transpose(F), b);
    const vec<real> direct = solve_transpose(F, b);
    for (idx i = 0; i < n; ++i) {
        EXPECT_EQ(via_view[i], direct[i]);
    }
    EXPECT_NEAR(rcond(F, A), rcond(lu(A), A), 1e-6 * rcond(lu(A), A));
}
