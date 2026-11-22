/// @file tests/test_condition.cpp
/// @brief Condition estimates against the exact \f$\|A^{-1}\|_1\f$ from an explicit inverse.

#include "linear/condition.hpp"
#include "linear/factorization/cholesky.hpp"
#include "linear/factorization/factor.hpp"
#include "linear/factorization/lu.hpp"
#include "linear/factorization/woodbury.hpp"
#include "linear/matrix_properties.hpp"
#include "linear/solvers/auto_linear.hpp"
#include <cmath>
#include <limits>
#include <gtest/gtest.h>
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

real exact_inverse_norm1(const mat<real> &A) {
    return opnorm1(inverse(lu(A)));
}

spmat sparse_of(const mat<real> &A) {
    array<idx> rows, columns;
    array<real> values;
    for (idx i = 0; i < A.rows(); ++i) {
        for (idx j = 0; j < A.cols(); ++j) {
            if (A(i, j) != 0.0) {
                rows.push_back(i);
                columns.push_back(j);
                values.push_back(A(i, j));
            }
        }
    }
    return spmat::from_triplets(A.rows(), A.cols(), rows, columns, values);
}

} // namespace

TEST(Condition, OpNorm1IsTheLargestColumnSum) {
    mat<real> A(2, 3, 0.0);
    A(0, 0) = 1.0, A(1, 0) = -2.0;
    A(0, 1) = 4.0;
    A(0, 2) = -1.0, A(1, 2) = 1.0;
    EXPECT_EQ(opnorm1(A), 4.0);
    EXPECT_EQ(opnorm1(sparse_of(A)), 4.0);
}

// A lower bound that is rarely far from the truth: never above it, and within a factor of 3.
TEST(Condition, EstimateBracketsTheExactInverseNorm) {
    for (unsigned seed = 1; seed <= 20; ++seed) {
        const idx n = 5 + seed;
        const mat<real> A = random_matrix(n, seed, seed % 4 == 0 ? 0.0 : 1.5);
        const real exact = exact_inverse_norm1(A);
        const real estimate = inverse_norm1_estimate(lu(A), n);
        EXPECT_LE(estimate, exact * (1.0 + 1e-10)) << "seed " << seed;
        EXPECT_GE(estimate, exact / 3.0) << "seed " << seed;
    }
}

TEST(Condition, ExactForADiagonalMatrix) {
    mat<real> A(4, 4, 0.0);
    A(0, 0) = 2.0, A(1, 1) = -0.01, A(2, 2) = 5.0, A(3, 3) = 1.0;
    EXPECT_NEAR(inverse_norm1_estimate(lu(A), 4), 100.0, 1e-10);
    EXPECT_NEAR(rcond(lu(A), A), 1.0 / (5.0 * 100.0), 1e-14);
}

TEST(Condition, HilbertMatrixIsIllConditioned) {
    const idx n = 10;
    mat<real> H(n, n, 0.0);
    for (idx i = 0; i < n; ++i) {
        for (idx j = 0; j < n; ++j) {
            H(i, j) = 1.0 / static_cast<real>(i + j + 1);
        }
    }
    // kappa_1 of the 10x10 Hilbert matrix is about 3.5e13.
    const real estimate = rcond(lu(H), H);
    EXPECT_GT(estimate, 1e-15);
    EXPECT_LT(estimate, 1e-12);
}

// The estimator is written once against `factorization`, so every factor type gets it.
TEST(Condition, EveryFactorizationGivesTheSameEstimate) {
    const idx n = 12;
    mat<real> A = random_matrix(n, 7, 0.0);
    mat<real> S(n, n, 0.0); // symmetric positive definite: A^T A + I
    for (idx i = 0; i < n; ++i) {
        for (idx j = 0; j < n; ++j) {
            for (idx k = 0; k < n; ++k) {
                S(i, j) += A(k, i) * A(k, j);
            }
        }
        S(i, i) += 1.0;
    }
    const real from_lu = rcond(lu(S), S);
    EXPECT_NEAR(rcond(cholesky(assume_spd(S)), S), from_lu, 1e-10 * from_lu);
    EXPECT_NEAR(rcond(auto_linear_solver(sparse_of(S)), sparse_of(S)), from_lu, 1e-10 * from_lu);
    EXPECT_NEAR(rcond(lu(A), A), 1.0 / (opnorm1(A) * inverse_norm1_estimate(lu(A), n)), 1e-15);
}

namespace {

vec<real> dense_times(const mat<real> &A, const vec<real> &x) {
    vec<real> y(A.rows(), 0.0);
    for (idx i = 0; i < A.rows(); ++i) {
        for (idx j = 0; j < A.cols(); ++j) {
            y[i] += A(i, j) * x[j];
        }
    }
    return y;
}

real norm_inf(const mat<real> &A) {
    real m = 0;
    for (idx i = 0; i < A.rows(); ++i) {
        real row = 0;
        for (idx j = 0; j < A.cols(); ++j) {
            row += std::abs(A(i, j));
        }
        m = std::max(m, row);
    }
    return m;
}

real norm_inf(const vec<real> &v) {
    real m = 0;
    for (idx i = 0; i < v.size(); ++i) {
        m = std::max(m, std::abs(v[i]));
    }
    return m;
}

real norm_1(const mat<real> &A) {
    real m = 0;
    for (idx j = 0; j < A.cols(); ++j) {
        real col = 0;
        for (idx i = 0; i < A.rows(); ++i) {
            col += std::abs(A(i, j));
        }
        m = std::max(m, col);
    }
    return m;
}

mat<real> hilbert(idx n) {
    mat<real> H(n, n, 0.0);
    for (idx i = 0; i < n; ++i) {
        for (idx j = 0; j < n; ++j) {
            H(i, j) = 1.0 / static_cast<real>(i + j + 1);
        }
    }
    return H;
}

long double binomial(long double n, long double k) {
    long double r = 1;
    for (long double i = 1; i <= k; ++i) {
        r = r * (n - k + i) / i;
    }
    return r;
}

mat<real> hilbert_inverse(idx n) {
    mat<real> X(n, n, 0.0);
    const auto N = static_cast<long double>(n);
    for (idx i = 1; i <= n; ++i) {
        for (idx j = 1; j <= n; ++j) {
            const auto I = static_cast<long double>(i);
            const auto J = static_cast<long double>(j);
            const long double c = binomial(I + J - 2, I - 1);
            const long double v = (I + J - 1) * binomial(N + I - 1, N - J) *
                                  binomial(N + J - 1, N - I) * c * c;
            X(i - 1, j - 1) = static_cast<real>(((i + j) % 2 == 0) ? v : -v);
        }
    }
    return X;
}

} // namespace

// The Hilbert matrix: a backward-stable solve leaves a tiny residual even though the forward
// error is as large as kappa * eps allows, and the rcond estimate must track the true kappa.
TEST(Condition, HilbertSolveIsBackwardStableAndRcondTracksKappa) {
    for (idx n : {4, 6, 8, 10, 12}) {
        const mat<real> H = hilbert(n);
        const real kappa = norm_1(H) * norm_1(hilbert_inverse(n));
        const auto f = lu(H);
        const real estimate = rcond(f, H);
        EXPECT_GT(estimate, 0.1 / kappa) << "n = " << n;
        EXPECT_LT(estimate, 10.0 / kappa) << "n = " << n;

        const vec<real> x_true(n, 1.0);
        const vec<real> b = dense_times(H, x_true);
        vec<real> x;
        solve(f, b, x);
        vec<real> r = dense_times(H, x);
        real error = 0;
        for (idx i = 0; i < n; ++i) {
            r[i] = b[i] - r[i];
            error = std::max(error, std::abs(x[i] - 1.0));
        }
        const real scale = norm_inf(H) * norm_inf(x);
        EXPECT_LE(norm_inf(r), 10.0 * static_cast<real>(n) * eps * scale) << "n = " << n;
        EXPECT_LE(error, 10.0 * static_cast<real>(n) * eps * kappa) << "n = " << n;
    }
}
