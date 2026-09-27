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

/// A retained factorization satisfying `num::retained_factorization`, backed by
/// a dense pivoted LU. The concept asks only for the four out-parameter solves.
class dense_base {
  public:
    explicit dense_base(const mat &A) : n_(A.rows()), factor_(lu(A)) {
        mat transposed = transpose(A);
        transpose_factor_ = lu(transposed);
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

/// Strictly diagonally dominant, so every matrix below is nonsingular and the
/// correction is never asked to rescue an ill-posed solve.
mat dominant(idx n, unsigned seed) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<real> entry(-1.0, 1.0);
    mat A(n, n, 0.0);
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

/// @brief Replace the listed rows and columns of `A`, and nothing else.
///
/// This is the change a swept subnetwork makes when it exchanges states between
/// successive sweeps, and it is exactly the precondition `low_rank_difference`
/// states: untouched rows keep every entry except where a changed column cuts
/// through them. Diagonals outside `changed` are therefore left alone. Entries
/// stay within the unit interval while `dominant` gives each diagonal a slack of
/// n, so replacing one off-diagonal per untouched row cannot cost dominance.
mat with_changed(const mat &A, view<const idx> changed, unsigned seed) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<real> entry(-1.0, 1.0);
    mat B = A;
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

vec random_vector(idx n, unsigned seed) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<real> entry(-1.0, 1.0);
    vec b(n, 0.0);
    for (idx i = 0; i < n; ++i) {
        b[i] = entry(generator);
    }
    return b;
}

constexpr real tolerance = 1e-9;

} // namespace

TEST(LowRankDifference, ReproducesTheChangedMatrix) {
    constexpr idx n = 9;
    const mat base = dominant(n, 11);
    const array<idx> changed{2, 5};
    const mat current = with_changed(base, changed, 12);

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
    const mat A = dominant(6, 21);
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
    const mat base = dominant(n, 41);
    const array<idx> changed{1, 4, 7};
    const mat current = with_changed(base, changed, 42);

    const dense_base retained(base);
    const woodbury_solver correction(
        retained, low_rank_difference(sparse_of(base), sparse_of(current), changed));
    EXPECT_EQ(correction.size(), n);
    EXPECT_EQ(correction.rank(), 2 * changed.size());

    const vec b = random_vector(n, 43);
    vec expected(n, 0.0);
    const lu_result fresh = lu(current);
    lu_solve(fresh, b, expected);

    const vec corrected = correction.solve(b);
    for (idx i = 0; i < n; ++i) {
        EXPECT_NEAR(corrected[i], expected[i], tolerance) << "entry " << i;
    }
}

TEST(WoodburySolver, MatchesAFreshTransposeFactorization) {
    constexpr idx n = 10;
    const mat base = dominant(n, 51);
    const array<idx> changed{0, 6};
    const mat current = with_changed(base, changed, 52);

    const dense_base retained(base);
    const woodbury_solver correction(
        retained, low_rank_difference(sparse_of(base), sparse_of(current), changed));

    const vec b = random_vector(n, 53);
    mat transposed = transpose(current);
    vec expected(n, 0.0);
    lu_solve(lu(transposed), b, expected);

    const vec corrected = correction.solve_transpose(b);
    for (idx i = 0; i < n; ++i) {
        EXPECT_NEAR(corrected[i], expected[i], tolerance) << "entry " << i;
    }
}

TEST(WoodburySolver, CorrectsSeveralRightHandSidesAtOnce) {
    constexpr idx n = 8;
    const mat base = dominant(n, 61);
    const array<idx> changed{3};
    const mat current = with_changed(base, changed, 62);

    const dense_base retained(base);
    const woodbury_solver correction(
        retained, low_rank_difference(sparse_of(base), sparse_of(current), changed));

    mat rhs(n, 3, 0.0);
    for (idx column = 0; column < 3; ++column) {
        const vec b = random_vector(n, 63 + static_cast<unsigned>(column));
        for (idx i = 0; i < n; ++i) {
            rhs(i, column) = b[i];
        }
    }
    mat expected(n, 3, 0.0);
    lu_solve(lu(current), rhs, expected);

    const mat corrected = correction.solve(rhs);
    for (idx i = 0; i < n; ++i) {
        for (idx column = 0; column < 3; ++column) {
            EXPECT_NEAR(corrected(i, column), expected(i, column), tolerance);
        }
    }
}

TEST(WoodburySolver, InverseDiagonalMatchesAFreshInverse) {
    constexpr idx n = 9;
    const mat base = dominant(n, 71);
    const array<idx> changed{2, 8};
    const mat current = with_changed(base, changed, 72);

    // The base diagonal by explicit solves, which is what a caller retains.
    vec base_diagonal(n, 0.0);
    const lu_result base_factor = lu(base);
    const lu_result current_factor = lu(current);
    vec expected_diagonal(n, 0.0);
    for (idx i = 0; i < n; ++i) {
        const vec e = unit_vector(n, i);
        vec column(n, 0.0);
        lu_solve(base_factor, e, column);
        base_diagonal[i] = column[i];
        lu_solve(current_factor, e, column);
        expected_diagonal[i] = column[i];
    }

    const dense_base retained(base);
    const woodbury_solver correction(
        retained, low_rank_difference(sparse_of(base), sparse_of(current), changed));
    const vec corrected = correction.inverse_diagonal(base_diagonal);

    for (idx i = 0; i < n; ++i) {
        EXPECT_NEAR(corrected[i], expected_diagonal[i], tolerance) << "entry " << i;
    }
}

TEST(WoodburySolver, RejectsAMismatchedBaseDiagonal) {
    const mat base = dominant(6, 81);
    const array<idx> changed{1};
    const mat current = with_changed(base, changed, 82);
    const dense_base retained(base);
    const woodbury_solver correction(
        retained, low_rank_difference(sparse_of(base), sparse_of(current), changed));
    const vec wrong(3, 1.0);
    EXPECT_THROW((void)correction.inverse_diagonal(wrong), std::invalid_argument);
}

TEST(UpdateInverseRows, MatchesRowsOfAFreshInverseAndItsSquare) {
    constexpr idx n = 8;
    const mat base = dominant(n, 91);
    const array<idx> changed{0, 5};
    const mat current = with_changed(base, changed, 92);
    const array<idx> carried{1, 4, 6};

    // Rows of the base inverse and its square, which the caller holds already.
    const lu_result base_factor = lu(base);
    mat first(carried.size(), n, 0.0), second(carried.size(), n, 0.0);
    for (idx k = 0; k < carried.size(); ++k) {
        const vec e = unit_vector(n, carried[k]);
        vec row(n, 0.0), row_squared(n, 0.0);
        mat transposed_base = transpose(base);
        const lu_result transposed_factor = lu(transposed_base);
        lu_solve(transposed_factor, e, row);
        lu_solve(transposed_factor, row, row_squared);
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
    mat transposed_current = transpose(current);
    const lu_result fresh = lu(transposed_current);
    for (idx k = 0; k < carried.size(); ++k) {
        const vec e = unit_vector(n, carried[k]);
        vec row(n, 0.0), row_squared(n, 0.0);
        lu_solve(fresh, e, row);
        lu_solve(fresh, row, row_squared);
        for (idx j = 0; j < n; ++j) {
            EXPECT_NEAR(first(k, j), row[j], tolerance) << "inverse row " << k << " entry " << j;
            EXPECT_NEAR(second(k, j), row_squared[j], tolerance)
                << "squared row " << k << " entry " << j;
        }
    }
}

TEST(SparseDiagonalSimilarity, AgreesWithTheDenseFormAndKeepsThePattern) {
    const mat A = dominant(7, 101);
    const spmat S = sparse_of(A);
    vec weights(7, 0.0);
    for (idx i = 0; i < 7; ++i) {
        weights[i] = 1.0 + (0.5 * static_cast<real>(i));
    }

    const spmat scaled = sparse_diagonal_similarity(S, weights);
    const mat reference = diagonal_similarity(S, weights);

    EXPECT_EQ(scaled.nnz(), S.nnz());
    const mat densified = dense(scaled);
    for (idx i = 0; i < 7; ++i) {
        for (idx j = 0; j < 7; ++j) {
            EXPECT_NEAR(densified(i, j), reference(i, j), tolerance);
        }
    }
}

TEST(SparseDiagonalSimilarity, PreservesTheSpectrumOnASymmetrizableMatrix) {
    // A similarity leaves the trace unchanged, whatever the weights.
    const mat A = dominant(6, 111);
    const spmat S = sparse_of(A);
    vec weights(6, 0.0);
    for (idx i = 0; i < 6; ++i) {
        weights[i] = 0.25 + static_cast<real>(i);
    }
    const mat scaled = dense(sparse_diagonal_similarity(S, weights));
    real original = 0.0, transformed = 0.0;
    for (idx i = 0; i < 6; ++i) {
        original += A(i, i);
        transformed += scaled(i, i);
    }
    EXPECT_NEAR(original, transformed, tolerance);
}

TEST(SparseDiagonalSimilarity, RejectsNonPositiveWeights) {
    const spmat S = sparse_of(dominant(4, 121));
    vec weights(4, 1.0);
    weights[2] = 0.0;
    EXPECT_THROW((void)sparse_diagonal_similarity(S, weights), std::invalid_argument);
}
