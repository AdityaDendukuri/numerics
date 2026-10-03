/// @file tests/test_woodbury.cpp
/// @brief Low-rank corrected solves against a retained factorization.
///
/// Every test here states the same claim in a different form: solving with the
/// corrected operator must match refactoring the changed matrix outright. The
/// reference is always a fresh `num::lu` of the current matrix, so a wrong
/// correction shows up as a diverging solution rather than as a plausible one.

#include "linear/factorization/lu.hpp"
#include "linear/factorization/woodbury.hpp"
#include "linear/matrix_properties.hpp"
#include "linear/matrix_utils.hpp"
#include "linear/sparse/sparse.hpp"
#include <gtest/gtest.h>
#include <random>

using namespace num;

namespace {

/// A retained factorization satisfying `num::factorization`, backed by
/// a dense pivoted LU. The concept asks only for the four out-parameter solves.
class dense_base {
  public:
    explicit dense_base(const mat<real> &A) : n_(A.rows()), factor_(lu(A)) {
        mat<real> transposed = transpose(A);
        transpose_factor_ = lu(transposed);
    }

    [[nodiscard]] idx size() const { return n_; }

    // Free functions found by argument-dependent lookup, as the library's own types do.
    template <class RHS>
    friend void solve(const dense_base &F, const RHS &rhs, RHS &out) {
        num::solve(F.factor_, rhs, out);
    }
    template <class RHS>
    friend void solve_transpose(const dense_base &F, const RHS &rhs, RHS &out) {
        num::solve(F.transpose_factor_, rhs, out);
    }

  private:
    idx n_;
    lu_result<real> factor_, transpose_factor_;
};

/// Strictly diagonally dominant, so every matrix below is nonsingular and the
/// correction is never asked to rescue an ill-posed solve.
mat<real> dominant(idx n, unsigned seed) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<real> entry(-1.0, 1.0);
    mat<real> A(n, n, 0.0);
    for (idx i = 0; i < n; ++i) {
        real off_diagonal = 0.0;
        for (idx j = 0; j < n; ++j) {
            if (i != j) {
                A(i, j) = entry(generator);
                off_diagonal += std::abs(A(i, j));
            }
        }
        A(i, i) = off_diagonal + static_cast<real>(n);
    }
    return A;
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

/// @brief Replace the listed rows and columns of `A`, and nothing else.
///
/// This is the change a swept subnetwork makes when it exchanges states between
/// successive sweeps, and it is exactly the precondition `low_rank_difference`
/// states: untouched rows keep every entry except where a changed column cuts
/// through them. Diagonals outside `changed` are therefore left alone. Entries
/// stay within the unit interval while `dominant` gives each diagonal a slack of
/// n, so replacing one off-diagonal per untouched row cannot cost dominance.
mat<real> with_changed(const mat<real> &A, view<const idx> changed, unsigned seed) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<real> entry(-1.0, 1.0);
    mat<real> B = A;
    for (idx k : changed) {
        real off_diagonal = 0.0;
        for (idx j = 0; j < B.cols(); ++j) {
            if (j != k) {
                B(k, j) = entry(generator);
                B(j, k) = entry(generator);
                off_diagonal += std::abs(B(k, j));
            }
        }
        B(k, k) = off_diagonal + static_cast<real>(B.rows());
    }
    return B;
}

vec<real> random_vector(idx n, unsigned seed) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<real> entry(-1.0, 1.0);
    vec<real> b(n, 0.0);
    for (idx i = 0; i < n; ++i) {
        b[i] = entry(generator);
    }
    return b;
}

constexpr real tolerance = 1e-9;

} // namespace

TEST(LowRankDifference, ReproducesTheChangedMatrix) {
    constexpr idx n = 9;
    const mat<real> base = dominant(n, 11);
    const array<idx> changed{2, 5};
    const mat<real> current = with_changed(base, changed, 12);

    const low_rank_update update = low_rank_difference(sparse_of(base), sparse_of(current), changed);
    ASSERT_EQ(update.left.cols(), 2 * changed.size());

    // base + P Q^T must be the current matrix entry for entry.
    for (idx i = 0; i < n; ++i) {
        for (idx j = 0; j < n; ++j) {
            real correction = 0.0;
            for (idx column = 0; column < update.left.cols(); ++column) {
                correction += update.left(i, column) * update.right(j, column);
            }
            EXPECT_NEAR(base(i, j) + correction, current(i, j), tolerance)
                << "entry (" << i << ", " << j << ")";
        }
    }
}

TEST(LowRankDifference, IsEmptyWhenNothingChanged) {
    const mat<real> A = dominant(6, 21);
    const spmat S = sparse_of(A);
    const low_rank_update update = low_rank_difference(S, S, {});
    EXPECT_EQ(update.left.cols(), 0);
}

TEST(LowRankDifference, RejectsAnIndexOutsideTheMatrix) {
    const spmat S = sparse_of(dominant(4, 31));
    const array<idx> changed{7};
    EXPECT_THROW((void)low_rank_difference(S, S, changed), std::out_of_range);
}

TEST(WoodburySolver, MatchesAFreshFactorization) {
    constexpr idx n = 10;
    const mat<real> base = dominant(n, 41);
    const array<idx> changed{1, 4, 7};
    const mat<real> current = with_changed(base, changed, 42);

    const dense_base retained(base);
    const woodbury_solver correction(
        retained, low_rank_difference(sparse_of(base), sparse_of(current), changed));
    EXPECT_EQ(correction.size(), n);
    EXPECT_EQ(correction.rank(), 2 * changed.size());

    const vec<real> b = random_vector(n, 43);
    vec<real> expected(n, 0.0);
    const lu_result<real> fresh = lu(current);
    solve(fresh, b, expected);

    const vec<real> corrected = solve(correction, b);
    for (idx i = 0; i < n; ++i) {
        EXPECT_NEAR(corrected[i], expected[i], tolerance) << "entry " << i;
    }
}

TEST(WoodburySolver, MatchesAFreshTransposeFactorization) {
    constexpr idx n = 10;
    const mat<real> base = dominant(n, 51);
    const array<idx> changed{0, 6};
    const mat<real> current = with_changed(base, changed, 52);

    const dense_base retained(base);
    const woodbury_solver correction(
        retained, low_rank_difference(sparse_of(base), sparse_of(current), changed));

    const vec<real> b = random_vector(n, 53);
    mat<real> transposed = transpose(current);
    vec<real> expected(n, 0.0);
    solve(lu(transposed), b, expected);

    const vec<real> corrected = solve_transpose(correction, b);
    for (idx i = 0; i < n; ++i) {
        EXPECT_NEAR(corrected[i], expected[i], tolerance) << "entry " << i;
    }
}

TEST(WoodburySolver, CorrectsSeveralRightHandSidesAtOnce) {
    constexpr idx n = 8;
    const mat<real> base = dominant(n, 61);
    const array<idx> changed{3};
    const mat<real> current = with_changed(base, changed, 62);

    const dense_base retained(base);
    const woodbury_solver correction(
        retained, low_rank_difference(sparse_of(base), sparse_of(current), changed));

    mat<real> rhs(n, 3, 0.0);
    for (idx column = 0; column < 3; ++column) {
        const vec<real> b = random_vector(n, 63 + static_cast<unsigned>(column));
        for (idx i = 0; i < n; ++i) {
            rhs(i, column) = b[i];
        }
    }
    mat<real> expected(n, 3, 0.0);
    solve(lu(current), rhs, expected);

    const mat<real> corrected = solve(correction, rhs);
    for (idx i = 0; i < n; ++i) {
        for (idx column = 0; column < 3; ++column) {
            EXPECT_NEAR(corrected(i, column), expected(i, column), tolerance);
        }
    }
}

TEST(WoodburySolver, InverseDiagonalMatchesAFreshInverse) {
    constexpr idx n = 9;
    const mat<real> base = dominant(n, 71);
    const array<idx> changed{2, 8};
    const mat<real> current = with_changed(base, changed, 72);

    // The base diagonal by explicit solves, which is what a caller retains.
    vec<real> base_diagonal(n, 0.0);
    const lu_result<real> base_factor = lu(base);
    const lu_result<real> current_factor = lu(current);
    vec<real> expected_diagonal(n, 0.0);
    for (idx i = 0; i < n; ++i) {
        const vec<real> e = unit_vector(n, i);
        vec<real> column(n, 0.0);
        solve(base_factor, e, column);
        base_diagonal[i] = column[i];
        solve(current_factor, e, column);
        expected_diagonal[i] = column[i];
    }

    const dense_base retained(base);
    const woodbury_solver correction(
        retained, low_rank_difference(sparse_of(base), sparse_of(current), changed));
    const vec<real> corrected = correction.inverse_diagonal(base_diagonal);

    for (idx i = 0; i < n; ++i) {
        EXPECT_NEAR(corrected[i], expected_diagonal[i], tolerance) << "entry " << i;
    }
}

TEST(WoodburySolver, RejectsAMismatchedBaseDiagonal) {
    const mat<real> base = dominant(6, 81);
    const array<idx> changed{1};
    const mat<real> current = with_changed(base, changed, 82);
    const dense_base retained(base);
    const woodbury_solver correction(
        retained, low_rank_difference(sparse_of(base), sparse_of(current), changed));
    const vec<real> wrong(3, 1.0);
    EXPECT_THROW((void)correction.inverse_diagonal(wrong), std::invalid_argument);
}

TEST(UpdateInverseRows, MatchesRowsOfAFreshInverseAndItsSquare) {
    constexpr idx n = 8;
    const mat<real> base = dominant(n, 91);
    const array<idx> changed{0, 5};
    const mat<real> current = with_changed(base, changed, 92);
    const array<idx> carried{1, 4, 6};

    // Rows of the base inverse and its square, which the caller holds already.
    const lu_result<real> base_factor = lu(base);
    mat<real> first(carried.size(), n, 0.0), second(carried.size(), n, 0.0);
    for (idx k = 0; k < carried.size(); ++k) {
        const vec<real> e = unit_vector(n, carried[k]);
        vec<real> row(n, 0.0), row_squared(n, 0.0);
        mat<real> transposed_base = transpose(base);
        const lu_result<real> transposed_factor = lu(transposed_base);
        solve(transposed_factor, e, row);
        solve(transposed_factor, row, row_squared);
        for (idx j = 0; j < n; ++j) {
            first(k, j) = row[j];
            second(k, j) = row_squared[j];
        }
    }

    const dense_base retained(base);
    const woodbury_solver correction(
        retained, low_rank_difference(sparse_of(base), sparse_of(current), changed));
    inverse_rows_workspace work;
    update_inverse_rows(correction, first, second, work);

    // The same rows from a fresh factorization of the changed matrix.
    mat<real> transposed_current = transpose(current);
    const lu_result<real> fresh = lu(transposed_current);
    for (idx k = 0; k < carried.size(); ++k) {
        const vec<real> e = unit_vector(n, carried[k]);
        vec<real> row(n, 0.0), row_squared(n, 0.0);
        solve(fresh, e, row);
        solve(fresh, row, row_squared);
        for (idx j = 0; j < n; ++j) {
            EXPECT_NEAR(first(k, j), row[j], tolerance) << "inverse row " << k << " entry " << j;
            EXPECT_NEAR(second(k, j), row_squared[j], tolerance)
                << "squared row " << k << " entry " << j;
        }
    }
}

TEST(SparseDiagonalSimilarity, AgreesWithTheDenseFormAndKeepsThePattern) {
    const mat<real> A = dominant(7, 101);
    const spmat S = sparse_of(A);
    vec<real> weights(7, 0.0);
    for (idx i = 0; i < 7; ++i) {
        weights[i] = 1.0 + (0.5 * static_cast<real>(i));
    }

    const spmat scaled = sparse_diagonal_similarity(S, weights);
    const mat<real> reference = diagonal_similarity(S, weights);

    EXPECT_EQ(scaled.nnz(), S.nnz());
    const mat<real> densified = dense(scaled);
    for (idx i = 0; i < 7; ++i) {
        for (idx j = 0; j < 7; ++j) {
            EXPECT_NEAR(densified(i, j), reference(i, j), tolerance);
        }
    }
}

TEST(SparseDiagonalSimilarity, PreservesTheSpectrumOnASymmetrizableMatrix) {
    // A similarity leaves the trace unchanged, whatever the weights.
    const mat<real> A = dominant(6, 111);
    const spmat S = sparse_of(A);
    vec<real> weights(6, 0.0);
    for (idx i = 0; i < 6; ++i) {
        weights[i] = 0.25 + static_cast<real>(i);
    }
    const mat<real> scaled = dense(sparse_diagonal_similarity(S, weights));
    real original = 0.0, transformed = 0.0;
    for (idx i = 0; i < 6; ++i) {
        original += A(i, i);
        transformed += scaled(i, i);
    }
    EXPECT_NEAR(original, transformed, tolerance);
}

TEST(SparseDiagonalSimilarity, RejectsNonPositiveWeights) {
    const spmat S = sparse_of(dominant(4, 121));
    vec<real> weights(4, 1.0);
    weights[2] = 0.0;
    EXPECT_THROW((void)sparse_diagonal_similarity(S, weights), std::invalid_argument);
}
