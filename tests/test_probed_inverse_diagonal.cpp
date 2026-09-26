/// @file tests/test_probed_inverse_diagonal.cpp
/// @brief Randomized diag(A^-1) against the exact diagonal.
///
/// The estimator is unbiased but random, so the reference here is always the
/// exact inverse diagonal obtained by one solve per index, and the assertions
/// are relative. A wrong scaling shows up as a systematic bias across every
/// entry rather than as noise in a few, which is why the tests check the worst
/// relative error rather than an average.

#include "linear/factorization/lu.hpp"
#include "linear/factorization/probed_inverse_diagonal.hpp"
#include "linear/matrix_properties.hpp"
#include "linear/matrix_utils.hpp"
#include "linear/sparse/sparse.hpp"
#include <gtest/gtest.h>
#include <cmath>

using namespace num;

namespace {

/// A retained factorization backed by dense pivoted LU, as in test_woodbury.
class dense_base {
  public:
    explicit dense_base(const mat &A) : n_(A.rows()), factor_(lu(assume_square(A))) {
        mat transposed = transpose(A);
        transpose_factor_ = lu(assume_square(transposed));
    }

    [[nodiscard]] idx size() const { return n_; }

    void solve(const vec &rhs, vec &out) const { lu_solve(factor_, rhs, out); }
    void solve(const mat &rhs, mat &out) const { lu_solve(factor_, rhs, out); }
    void solve_transpose(const vec &rhs, vec &out) const { lu_solve(transpose_factor_, rhs, out); }
    void solve_transpose(const mat &rhs, mat &out) const { lu_solve(transpose_factor_, rhs, out); }

  private:
    idx n_;
    lu_result factor_, transpose_factor_;
};

spmat sparse_of(const mat &A) {
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

/// @brief A truncated rate matrix on a path graph with a leak at each end.
///
/// The sign convention is the M-matrix one: negative off-diagonals, positive
/// diagonal, and a row sum that is positive wherever probability can leave. That
/// makes it a nonsingular M-matrix, which is what the estimator assumes.
mat path_rate_matrix(idx n, real rate, real leak) {
    mat A(n, n, 0.0);
    for (idx i = 0; i < n; ++i) {
        real total = 0.0;
        if (i > 0) {
            A(i, i - 1) = -rate;
            total += rate;
        }
        if (i + 1 < n) {
            A(i, i + 1) = -rate;
            total += rate;
        }
        A(i, i) = total + leak;
    }
    return A;
}

/// @brief The same chain made reversible, with its stationary measure.
///
/// Birth rate `up` and death rate `down` give detailed balance with
/// \f$\pi_i \propto (up/down)^i\f$, so the diagonal similarity through
/// \f$\sqrt{\pi}\f$ symmetrizes the matrix and the estimator may take its
/// reversible path.
mat birth_death_rate_matrix(idx n, real up, real down, real leak, vec &stationary) {
    mat A(n, n, 0.0);
    stationary = vec(n, 0.0);
    real weight = 1.0;
    for (idx i = 0; i < n; ++i) {
        stationary[i] = weight;
        weight *= up / down;
    }
    for (idx i = 0; i < n; ++i) {
        real total = 0.0;
        if (i > 0) {
            A(i, i - 1) = -down;
            total += down;
        }
        if (i + 1 < n) {
            A(i, i + 1) = -up;
            total += up;
        }
        A(i, i) = total + leak;
    }
    return A;
}

/// diag(A^-1) by one solve per index, which is what the estimator replaces.
vec exact_inverse_diagonal(const mat &A) {
    const idx n = A.rows();
    const lu_result factor = lu(assume_square(A));
    vec diagonal(n, 0.0);
    for (idx i = 0; i < n; ++i) {
        const vec e = unit_vector(n, i);
        vec column(n, 0.0);
        lu_solve(factor, e, column);
        diagonal[i] = column[i];
    }
    return diagonal;
}

real worst_relative_error(const vec &estimate, const vec &exact) {
    real worst = 0.0;
    for (idx i = 0; i < exact.size(); ++i) {
        worst = std::max(worst, std::abs(estimate[i] - exact[i]) / std::abs(exact[i]));
    }
    return worst;
}

} // namespace

TEST(ProbedInverseDiagonal, ApproximatesTheExactDiagonalOnANonReversibleChain) {
    constexpr idx n = 24;
    const mat A = path_rate_matrix(n, 1.0, 0.35);
    const vec exact = exact_inverse_diagonal(A);

    const dense_base retained(A);
    const vec estimate = inverse_diagonal(retained, sparse_of(A), {},
                                                 {.probes = 600, .seed = 7});

    ASSERT_EQ(estimate.size(), n);
    // The estimate is a chi-square mean over 600 probes, so its relative
    // standard deviation is about sqrt(2/600), near 6%. Allow a few of those.
    EXPECT_LT(worst_relative_error(estimate, exact), 0.25);
}

TEST(ProbedInverseDiagonal, ApproximatesTheExactDiagonalOnAReversibleChain) {
    constexpr idx n = 20;
    vec stationary;
    const mat A = birth_death_rate_matrix(n, 1.0, 1.6, 0.3, stationary);
    const vec exact = exact_inverse_diagonal(A);

    vec symmetrizer(n, 0.0);
    for (idx i = 0; i < n; ++i) {
        symmetrizer[i] = std::sqrt(stationary[i]);
    }

    const dense_base retained(A);
    const vec estimate = inverse_diagonal(retained, sparse_of(A), symmetrizer,
                                                 {.probes = 600, .seed = 11});

    ASSERT_EQ(estimate.size(), n);
    EXPECT_LT(worst_relative_error(estimate, exact), 0.25);
}

TEST(ProbedInverseDiagonal, ReversibleAndGeneralPathsAgreeOnAReversibleChain) {
    // The same matrix through both branches: supplying sqrt(pi) only changes how
    // the scaling is found, never what is estimated.
    constexpr idx n = 16;
    vec stationary;
    const mat A = birth_death_rate_matrix(n, 1.0, 1.4, 0.4, stationary);
    vec symmetrizer(n, 0.0);
    for (idx i = 0; i < n; ++i) {
        symmetrizer[i] = std::sqrt(stationary[i]);
    }

    const dense_base retained(A);
    const spmat S = sparse_of(A);
    const vec with_measure =
        inverse_diagonal(retained, S, symmetrizer, {.probes = 800, .seed = 5});
    const vec without_measure = inverse_diagonal(retained, S, {}, {.probes = 800, .seed = 5});

    const vec exact = exact_inverse_diagonal(A);
    EXPECT_LT(worst_relative_error(with_measure, exact), 0.25);
    EXPECT_LT(worst_relative_error(without_measure, exact), 0.25);
}

TEST(ProbedInverseDiagonal, EveryEstimateIsPositive) {
    // A squared row norm cannot be negative, and the cut-time score divides by
    // it, so a nonpositive entry would be a defect rather than sampling noise.
    constexpr idx n = 18;
    const mat A = path_rate_matrix(n, 1.0, 0.2);
    const dense_base retained(A);
    const vec estimate =
        inverse_diagonal(retained, sparse_of(A), {}, {.probes = 8, .seed = 3});
    for (idx i = 0; i < n; ++i) {
        EXPECT_GT(estimate[i], 0.0) << "entry " << i;
    }
}

TEST(ProbedInverseDiagonal, SharpensAsProbesAreAdded) {
    constexpr idx n = 20;
    const mat A = path_rate_matrix(n, 1.0, 0.3);
    const vec exact = exact_inverse_diagonal(A);
    const dense_base retained(A);
    const spmat S = sparse_of(A);

    const vec few = inverse_diagonal(retained, S, {}, {.probes = 4, .seed = 17});
    const vec many = inverse_diagonal(retained, S, {}, {.probes = 1500, .seed = 17});

    EXPECT_LT(worst_relative_error(many, exact), worst_relative_error(few, exact));
}

TEST(ProbedInverseDiagonal, RejectsAnEmptyProbeBlock) {
    const mat A = path_rate_matrix(6, 1.0, 0.5);
    const dense_base retained(A);
    EXPECT_THROW((void)inverse_diagonal(retained, sparse_of(A), {}, {.probes = 0}),
                 std::invalid_argument);
}

TEST(ProbedInverseDiagonal, RejectsAMismatchedSymmetrizer) {
    const mat A = path_rate_matrix(6, 1.0, 0.5);
    const dense_base retained(A);
    const vec wrong(3, 1.0);
    EXPECT_THROW((void)inverse_diagonal(retained, sparse_of(A), wrong),
                 std::invalid_argument);
}
