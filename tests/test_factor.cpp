/// @file tests/test_factor.cpp
/// @brief Concrete dense, block, and sparse factorizations share the solve protocol.

#include "linear/factorization/cholesky.hpp"
#include "linear/factorization/factor.hpp"
#include "linear/factorization/lu.hpp"
#include "linear/factorization/probed_inverse_diagonal.hpp"
#include "linear/factorization/reuse.hpp"
#include "linear/factorization/woodbury.hpp"
#include "linear/graph/levels.hpp"
#include "linear/matrix_properties.hpp"
#include "linear/matrix_utils.hpp"
#include "linear/solvers/auto_linear.hpp"
#include "linear/solvers/cg.hpp"
#include "linear/sparse/klu.hpp"
#include "linear/sparse/sparse.hpp"
#include "linear/subspace.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <vector>

using namespace num;

namespace {

constexpr real tolerance = 1e-9;

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

mat<real> path_rate_matrix(idx n, real up, real down, real leak) {
    mat<real> R(n, n, 0.0);
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

vec<real> path_symmetrizer(idx n, real up, real down) {
    vec<real> h(n, 0.0);
    real pi = 1.0;
    for (idx i = 0; i < n; ++i) {
        h[i] = std::sqrt(pi);
        pi *= up / down;
    }
    return h;
}

vec<real> reference_solve(const mat<real> &R, const vec<real> &b, bool transposed) {
    mat<real> A = transposed ? mat<real>(transpose(R)) : R;
    vec<real> x(b.size(), 0.0);
    solve(lu(A), b, x);
    return x;
}

vec<real> exact_inverse_diagonal(const mat<real> &R) {
    const idx n = R.rows();
    const lu_result<real> Z = lu(R);
    vec<real> diagonal(n, 0.0);
    for (idx i = 0; i < n; ++i) {
        vec<real> column(n, 0.0);
        solve(Z, unit_vector(n, i), column);
        diagonal[i] = column[i];
    }
    return diagonal;
}

vec<real> ramp(idx n, unsigned offset) {
    vec<real> b(n, 0.0);
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
void expect_solves(const F &Z, const mat<real> &R, const vec<real> &b) {
    const vec<real> x = solve(Z, b);
    const vec<real> y = solve(transpose(Z), b);
    const vec<real> expected_x = reference_solve(R, b, false);
    const vec<real> expected_y = reference_solve(R, b, true);
    for (idx i = 0; i < b.size(); ++i) {
        EXPECT_NEAR(x[i], expected_x[i], tolerance) << "forward entry " << i;
        EXPECT_NEAR(y[i], expected_y[i], tolerance) << "transposed entry " << i;
    }
}

} // namespace

TEST(Factor, DenseBlockAndSparseSolveTheSameSystem) {
    constexpr idx n = 12;
    const mat<real> R = path_rate_matrix(n, 1.3, 0.7, 0.3);
    const spmat S = sparse_of(R);
    const vec<real> b = ramp(n, 5);
    const array<idx> levels = natural_levels(n);

    const auto Z_dense = lu(S, no_pivot);
    const auto Z_block = lu(S, blocks(levels));
    const auto Z_sparse = lu(S, sparse);
    expect_solves(Z_dense, R, b);
    expect_solves(Z_block, R, b);
    expect_solves(Z_sparse, R, b);
}

TEST(Factor, DiagonalSimilarityUsesCholeskyWithoutChangingTheAnswer) {
    constexpr idx n = 10;
    constexpr real up = 1.0, down = 1.5;
    const mat<real> R = path_rate_matrix(n, up, down, 0.25);
    const spmat S = sparse_of(R);
    const vec<real> h = path_symmetrizer(n, up, down);
    const vec<real> b = ramp(n, 2);
    const array<idx> levels = natural_levels(n);

    const auto Z_dense = cholesky(S, h);
    const auto Z_block = cholesky(S, blocks(levels), h);
    expect_solves(Z_dense, R, b);
    expect_solves(Z_block, R, b);
}

TEST(Factor, SolvesSeveralRightHandSidesAndAcceptsAnOutputBuffer) {
    constexpr idx n = 9;
    const mat<real> R = path_rate_matrix(n, 1.0, 0.8, 0.5);
    const auto Z = lu(sparse_of(R), no_pivot);
    mat<real> B(n, 3, 0.0);
    for (idx column = 0; column < B.cols(); ++column) {
        const vec<real> b = ramp(n, static_cast<unsigned>(column));
        for (idx i = 0; i < n; ++i) {
            B(i, column) = b[i];
        }
    }

    mat<real> X;
    solve(Z, B, X);
    for (idx column = 0; column < B.cols(); ++column) {
        vec<real> b(n, 0.0);
        for (idx i = 0; i < n; ++i) {
            b[i] = B(i, column);
        }
        const vec<real> expected = reference_solve(R, b, false);
        for (idx i = 0; i < n; ++i) {
            EXPECT_NEAR(X(i, column), expected[i], tolerance);
        }
    }
}

TEST(Factor, ExactInverseDiagonalUsesTheStoredFactor) {
    constexpr idx n = 11;
    const mat<real> R = path_rate_matrix(n, 1.0, 1.2, 0.35);
    const spmat S = sparse_of(R);
    const array<idx> levels = natural_levels(n);
    const vec<real> expected = exact_inverse_diagonal(R);

    const auto Z_dense = lu(S, no_pivot);
    const auto Z_block = lu(S, blocks(levels));
    for (const vec<real> diagonal : {inverse_diagonal(Z_dense, n), inverse_diagonal(Z_block, n)}) {
        for (idx i = 0; i < n; ++i) {
            EXPECT_NEAR(diagonal[i], expected[i], tolerance) << "entry " << i;
        }
    }
}

TEST(Factor, SparseInverseDiagonalIsProbedExplicitly) {
    constexpr idx n = 20;
    const mat<real> R = path_rate_matrix(n, 1.0, 1.0, 0.3);
    const spmat S = sparse_of(R);
    const auto Z = lu(S, sparse);
    const vec<real> diagonal =
        inverse_diagonal(Z, S, {}, inverse_diagonal_options{.probes = 800, .seed = 9});
    const vec<real> expected = exact_inverse_diagonal(R);

    real worst = 0.0;
    for (idx i = 0; i < n; ++i) {
        EXPECT_GT(diagonal[i], 0.0);
        worst = std::max(worst, std::abs(diagonal[i] - expected[i]) / expected[i]);
    }
    EXPECT_LT(worst, 0.25);
}

TEST(Factor, GraphDistanceLevelsFeedTheBlockFactorDirectly) {
    constexpr idx n = 14;
    const mat<real> R = path_rate_matrix(n, 1.0, 1.0, 0.45);
    const spmat S = sparse_of(R);
    const array<idx> levels = graph_distance_levels(S, idx{0});
    const auto Z = lu(S, blocks(levels));
    expect_solves(Z, R, ramp(n, 6));
}

TEST(Factor, RefactoringASuffixMatchesAFreshFactorization) {
    constexpr idx n = 12;
    const mat<real> old_R = path_rate_matrix(n, 1.0, 1.0, 0.4);
    const array<idx> levels = natural_levels(n);
    const suffix_block_lu Z = lu(sparse_of(old_R), blocks(levels), nullptr, {});

    mat<real> R = old_R;
    R(9, 9) += 0.75;
    const array<idx> changed{9};
    const suffix_block_lu updated = lu(sparse_of(R), blocks(levels), &Z, changed);

    EXPECT_TRUE(updated.reused());
    EXPECT_GT(updated.reused_blocks, 0);
    EXPECT_LE(updated.reused_blocks, 9);
    expect_solves(updated.factor, R, ramp(n, 8));
}

TEST(Factor, ReusableBlockCholeskyMatchesTheNonsymmetricSystem) {
    constexpr idx n = 12;
    constexpr real up = 1.2, down = 0.8;
    const array<idx> levels = natural_levels(n);
    vec<real> h(n, 1.0);
    for (idx i = 1; i < n; ++i) {
        h[i] = h[i - 1] * std::sqrt(up / down);
    }

    const mat<real> old_R = path_rate_matrix(n, up, down, 0.4);
    const suffix_block_cholesky Z = cholesky(sparse_of(old_R), blocks(levels), h, nullptr, {});

    mat<real> R = old_R;
    R(9, 9) += 0.75;
    const array<idx> changed{9};
    const suffix_block_cholesky updated = cholesky(sparse_of(R), blocks(levels), h, &Z, changed);

    EXPECT_TRUE(updated.reused());
    EXPECT_GT(updated.reused_blocks, 0);
    expect_solves(updated, R, ramp(n, 8));
}

TEST(Factor, ConcreteFactorsAreCorrectableByWoodbury) {
    constexpr idx n = 10;
    const mat<real> old_R = path_rate_matrix(n, 1.0, 1.0, 0.4);
    mat<real> R = old_R;
    R(4, 4) += 0.5;
    R(7, 7) += 0.25;
    const array<idx> changed{4, 7};
    const auto Z = lu(sparse_of(old_R), no_pivot);
    const woodbury_solver correction(Z,
                                     low_rank_difference(sparse_of(old_R), sparse_of(R), changed));

    const vec<real> b = ramp(n, 3);
    const vec<real> x = solve(correction, b);
    const vec<real> expected = reference_solve(R, b, false);
    for (idx i = 0; i < n; ++i) {
        EXPECT_NEAR(x[i], expected[i], tolerance);
    }

    const vec<real> corrected = correction.inverse_diagonal(inverse_diagonal(Z, n));
    const vec<real> expected_diagonal = exact_inverse_diagonal(R);
    for (idx i = 0; i < n; ++i) {
        EXPECT_NEAR(corrected[i], expected_diagonal[i], tolerance);
    }
}

TEST(Factor, RejectsAnInvalidBlockLabellingAndNonPositiveSimilarityWeights) {
    const spmat R = sparse_of(path_rate_matrix(6, 1.0, 1.0, 0.5));
    const array<idx> short_levels{0, 1, 2};
    EXPECT_THROW(lu(R, blocks(short_levels)), std::invalid_argument);

    vec<real> h(6, 1.0);
    h[2] = -1.0;
    EXPECT_THROW(cholesky(R, h), std::invalid_argument);
}

// Every factorization that can solve with A^T models one concept, so generic code such as
// woodbury_solver, transpose(F) and probed inverse diagonals takes any of them.
static_assert(factorization<lu_result<real>>);
static_assert(factorization<cholesky_result>);
static_assert(factorization<block_lu_factor>);
static_assert(factorization<block_cholesky_factor>);
static_assert(factorization<similar_factor<cholesky_result>>);
static_assert(factorization<auto_linear_solver>);
static_assert(factorization<klu_factorization>);
static_assert(factorization<corrected_lu>);
static_assert(factorization<suffix_block_lu>);
static_assert(factorization<suffix_block_cholesky>);
static_assert(factorization<woodbury_solver<lu_result<real>>>);
static_assert(factorization<detail::transposed_factor<lu_result<real>>>);

// The raw-pointer overloads, on the caller's own storage.

TEST(RawPointer, CholeskyAndLuAgreeOnAnSpdSystem) {
    const idx n = 3;
    std::vector<real> A{4, 1, 0, 1, 3, 1, 0, 1, 5}, L(n * n), b{1, 2, 3}, x(n), Ax(n);
    ASSERT_TRUE(cholesky(L.data(), A.data(), n));
    cholesky_solve(x.data(), L.data(), b.data(), n);
    kernel::matvec(Ax.data(), A.data(), x.data(), n, n);
    for (idx i = 0; i < n; ++i)
        EXPECT_NEAR(Ax[i], b[i], 1e-12);

    std::vector<real> blocked = A;
    ASSERT_TRUE(cholesky_blocked(blocked.data(), n, 2));
    for (idx i = 0; i < n; ++i)
        for (idx j = 0; j <= i; ++j)
            EXPECT_NEAR(blocked[(i * n) + j], L[(i * n) + j], 1e-12);

    std::vector<real> LU = A, x_lu(n);
    std::vector<idx> pivots(n);
    ASSERT_TRUE(lu_factor(LU.data(), pivots.data(), n));
    lu_solve(x_lu.data(), LU.data(), pivots.data(), b.data(), n);
    for (idx i = 0; i < n; ++i)
        EXPECT_NEAR(x_lu[i], x[i], 1e-12);

    std::vector<real> blocked_lu = A, x_blocked(n);
    std::vector<idx> blocked_pivots(n);
    ASSERT_TRUE(lu_factor_blocked(blocked_lu.data(), blocked_pivots.data(), n, 2));
    lu_solve(x_blocked.data(), blocked_lu.data(), blocked_pivots.data(), b.data(), n);
    for (idx i = 0; i < n; ++i)
        EXPECT_NEAR(x_blocked[i], x[i], 1e-12);
}

TEST(RawPointer, SolvesMayOverwriteTheirInput) {
    const idx n = 3;
    const std::vector<real> A{4, 1, 0, 1, 3, 1, 0, 1, 5}, b{1, 2, 3};
    std::vector<real> L(n * n), x(n);
    ASSERT_TRUE(cholesky(L.data(), A.data(), n));
    cholesky_solve(x.data(), L.data(), b.data(), n);

    std::vector<real> in_place = A, rhs = b;
    ASSERT_TRUE(cholesky(in_place.data(), in_place.data(), n));
    cholesky_solve(rhs.data(), in_place.data(), rhs.data(), n);
    for (idx i = 0; i < n; ++i)
        EXPECT_NEAR(rhs[i], x[i], 1e-12);

    std::vector<real> LU = A, lu_rhs = b;
    std::vector<idx> pivots(n);
    ASSERT_TRUE(lu_factor(LU.data(), pivots.data(), n));
    lu_solve(lu_rhs.data(), LU.data(), pivots.data(), lu_rhs.data(), n);
    for (idx i = 0; i < n; ++i)
        EXPECT_NEAR(lu_rhs[i], x[i], 1e-12);
}

TEST(RawPointer, ModifiedGramSchmidtRemovesTheBasisComponents) {
    const std::vector<real> basis{1, 0, 0, 1, 0, 0};
    std::vector<real> v{2, 3, 4};
    mgs_columns(v.data(), basis.data(), 2, 3, 2);
    EXPECT_NEAR(v[0], 0.0, 1e-12);
    EXPECT_NEAR(v[1], 0.0, 1e-12);
    EXPECT_NEAR(v[2], 4.0, 1e-12);
}

TEST(RawPointer, CgConvergesOnAMatrixFreeLaplacian) {
    const idx n = 64;
    auto A = [&](const real *v, real *out) {
        for (idx i = 0; i < n; ++i) {
            real s = 2.1 * v[i];
            if (i > 0)
                s -= v[i - 1];
            if (i + 1 < n)
                s -= v[i + 1];
            out[i] = s;
        }
    };
    std::vector<real> b(n, 1.0), x(n, 0.0), work(3 * n), Ax(n);
    const auto r = cg(A, x.data(), b.data(), n, work.data(), 1e-12, 500);
    ASSERT_TRUE(r.converged);
    A(x.data(), Ax.data());
    for (idx i = 0; i < n; ++i)
        EXPECT_NEAR(Ax[i], b[i], 1e-9);
}
