/// @file tests/test_factor.cpp
/// @brief Concrete dense, block, and sparse factorizations share the solve protocol.

#include "linear/factorization/factor.hpp"
#include "linear/factorization/lu.hpp"
#include "linear/factorization/probed_inverse_diagonal.hpp"
#include "linear/factorization/woodbury.hpp"
#include "linear/graph/levels.hpp"
#include "linear/matrix_properties.hpp"
#include "linear/matrix_utils.hpp"
#include "linear/sparse/sparse.hpp"
#include <cmath>
#include <gtest/gtest.h>

using namespace num;

namespace {

constexpr real tolerance = 1e-9;

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

mat path_rate_matrix(idx n, real up, real down, real leak) {
    mat R(n, n, 0.0);
    for (idx i = 0; i < n; ++i) {
        real total = 0.0;
        if (i > 0) {
            R(i, i - 1) = -down;
            total += down;
        }
        if (i + 1 < n) {
            R(i, i + 1) = -up;
            total += up;
        }
        R(i, i) = total + leak;
    }
    return R;
}

vec path_symmetrizer(idx n, real up, real down) {
    vec h(n, 0.0);
    real pi = 1.0;
    for (idx i = 0; i < n; ++i) {
        h[i] = std::sqrt(pi);
        pi *= up / down;
    }
    return h;
}

vec reference_solve(const mat &R, const vec &b, bool transposed) {
    mat A = transposed ? mat(transpose(R)) : R;
    vec x(b.size(), 0.0);
    lu_solve(lu(A), b, x);
    return x;
}

vec exact_inverse_diagonal(const mat &R) {
    const idx n = R.rows();
    const lu_result Z = lu(R);
    vec diagonal(n, 0.0);
    for (idx i = 0; i < n; ++i) {
        vec column(n, 0.0);
        lu_solve(Z, unit_vector(n, i), column);
        diagonal[i] = column[i];
    }
    return diagonal;
}

vec ramp(idx n, unsigned offset) {
    vec b(n, 0.0);
    for (idx i = 0; i < n; ++i) {
        b[i] = 1.0 + std::sin(static_cast<real>(i + offset));
    }
    return b;
}

array<idx> natural_levels(idx n) {
    array<idx> levels(n);
    for (idx i = 0; i < n; ++i) {
        levels[i] = i;
    }
    return levels;
}

template <class F>
void expect_solves(const F &Z, const mat &R, const vec &b) {
    const vec x = solve(Z, b);
    const vec y = solve(transpose(Z), b);
    const vec expected_x = reference_solve(R, b, false);
    const vec expected_y = reference_solve(R, b, true);
    for (idx i = 0; i < b.size(); ++i) {
        EXPECT_NEAR(x[i], expected_x[i], tolerance) << "forward entry " << i;
        EXPECT_NEAR(y[i], expected_y[i], tolerance) << "transposed entry " << i;
    }
}

} // namespace

TEST(Factor, DenseBlockAndSparseSolveTheSameSystem) {
    constexpr idx n = 12;
    const mat R = path_rate_matrix(n, 1.3, 0.7, 0.3);
    const spmat S = sparse_of(R);
    const vec b = ramp(n, 5);
    const array<idx> levels = natural_levels(n);

    const auto Z_dense = factor(S);
    const auto Z_block = factor(S, blocks(levels));
    const auto Z_sparse = factor(S, sparse);
    expect_solves(Z_dense, R, b);
    expect_solves(Z_block, R, b);
    expect_solves(Z_sparse, R, b);
}

TEST(Factor, DiagonalSimilarityUsesCholeskyWithoutChangingTheAnswer) {
    constexpr idx n = 10;
    constexpr real up = 1.0, down = 1.5;
    const mat R = path_rate_matrix(n, up, down, 0.25);
    const spmat S = sparse_of(R);
    const vec h = path_symmetrizer(n, up, down);
    const vec b = ramp(n, 2);
    const array<idx> levels = natural_levels(n);

    const auto Z_dense = factor(S, h);
    const auto Z_block = factor(S, blocks(levels), h);
    expect_solves(Z_dense, R, b);
    expect_solves(Z_block, R, b);
}

TEST(Factor, SolvesSeveralRightHandSidesAndAcceptsAnOutputBuffer) {
    constexpr idx n = 9;
    const mat R = path_rate_matrix(n, 1.0, 0.8, 0.5);
    const auto Z = factor(sparse_of(R));
    mat B(n, 3, 0.0);
    for (idx column = 0; column < B.cols(); ++column) {
        const vec b = ramp(n, static_cast<unsigned>(column));
        for (idx i = 0; i < n; ++i) {
            B(i, column) = b[i];
        }
    }

    mat X;
    solve(Z, B, X);
    for (idx column = 0; column < B.cols(); ++column) {
        vec b(n, 0.0);
        for (idx i = 0; i < n; ++i) {
            b[i] = B(i, column);
        }
        const vec expected = reference_solve(R, b, false);
        for (idx i = 0; i < n; ++i) {
            EXPECT_NEAR(X(i, column), expected[i], tolerance);
        }
    }
}

TEST(Factor, ExactInverseDiagonalUsesTheStoredFactor) {
    constexpr idx n = 11;
    const mat R = path_rate_matrix(n, 1.0, 1.2, 0.35);
    const spmat S = sparse_of(R);
    const array<idx> levels = natural_levels(n);
    const vec expected = exact_inverse_diagonal(R);

    const auto Z_dense = factor(S);
    const auto Z_block = factor(S, blocks(levels));
    for (const vec diagonal : {inverse_diagonal(Z_dense, n), inverse_diagonal(Z_block, n)}) {
        for (idx i = 0; i < n; ++i) {
            EXPECT_NEAR(diagonal[i], expected[i], tolerance) << "entry " << i;
        }
    }
}

TEST(Factor, SparseInverseDiagonalIsProbedExplicitly) {
    constexpr idx n = 20;
    const mat R = path_rate_matrix(n, 1.0, 1.0, 0.3);
    const spmat S = sparse_of(R);
    const auto Z = factor(S, sparse);
    const vec diagonal =
        inverse_diagonal(Z, S, {}, inverse_diagonal_options{.probes = 800, .seed = 9});
    const vec expected = exact_inverse_diagonal(R);

    real worst = 0.0;
    for (idx i = 0; i < n; ++i) {
        EXPECT_GT(diagonal[i], 0.0);
        worst = std::max(worst, std::abs(diagonal[i] - expected[i]) / expected[i]);
    }
    EXPECT_LT(worst, 0.25);
}

TEST(Factor, GraphDistanceLevelsFeedTheBlockFactorDirectly) {
    constexpr idx n = 14;
    const mat R = path_rate_matrix(n, 1.0, 1.0, 0.45);
    const spmat S = sparse_of(R);
    const array<idx> levels = graph_distance_levels(S, idx{0});
    const auto Z = factor(S, blocks(levels));
    expect_solves(Z, R, ramp(n, 6));
}

TEST(Factor, RefactoringASuffixMatchesAFreshFactorization) {
    constexpr idx n = 12;
    const mat old_R = path_rate_matrix(n, 1.0, 1.0, 0.4);
    const array<idx> levels = natural_levels(n);
    const auto Z = factor(sparse_of(old_R), blocks(levels));

    mat R = old_R;
    R(9, 9) += 0.75;
    const array<idx> changed{9};
    suffix_reuse_report report;
    const auto updated = refactor_suffix(Z, sparse_of(R), blocks(levels), changed, &report);

    ASSERT_TRUE(updated.has_value());
    EXPECT_EQ(report.blocks, n);
    EXPECT_GT(report.reused_blocks, 0);
    EXPECT_EQ(report.reused_rows, report.reused_blocks);
    expect_solves(*updated, R, ramp(n, 8));
}

TEST(Factor, ConcreteFactorsAreCorrectableByWoodbury) {
    constexpr idx n = 10;
    const mat old_R = path_rate_matrix(n, 1.0, 1.0, 0.4);
    mat R = old_R;
    R(4, 4) += 0.5;
    R(7, 7) += 0.25;
    const array<idx> changed{4, 7};
    const auto Z = factor(sparse_of(old_R));
    const woodbury_solver correction(Z,
                                     low_rank_difference(sparse_of(old_R), sparse_of(R), changed));

    const vec b = ramp(n, 3);
    const vec x = correction.solve(b);
    const vec expected = reference_solve(R, b, false);
    for (idx i = 0; i < n; ++i) {
        EXPECT_NEAR(x[i], expected[i], tolerance);
    }

    const vec corrected = correction.inverse_diagonal(inverse_diagonal(Z, n));
    const vec expected_diagonal = exact_inverse_diagonal(R);
    for (idx i = 0; i < n; ++i) {
        EXPECT_NEAR(corrected[i], expected_diagonal[i], tolerance);
    }
}

TEST(Factor, RejectsAnInvalidBlockLabellingAndNonPositiveSimilarityWeights) {
    const spmat R = sparse_of(path_rate_matrix(6, 1.0, 1.0, 0.5));
    const array<idx> short_levels{0, 1, 2};
    EXPECT_THROW(factor(R, blocks(short_levels)), std::invalid_argument);

    vec h(6, 1.0);
    h[2] = -1.0;
    EXPECT_THROW(factor(R, h), std::invalid_argument);
}
