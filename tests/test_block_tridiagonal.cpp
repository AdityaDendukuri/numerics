/// @file tests/test_block_tridiagonal.cpp
/// @brief Block LU and block Cholesky against dense factorization.
///
/// Every numerical test compares against the dense factorization of the same
/// matrix, because that is the only reference that pins down both the block
/// elimination and the permutation at once. A block algorithm with a correct
/// elimination and an inverted permutation still solves *a* system exactly, just
/// not the one it was handed, and a residual check alone would not catch it —
/// the tests below compare solutions entrywise against `num::lu` on the dense
/// matrix, so a wrong ordering shows up immediately.

#include "linear/factorization/block_tridiagonal.hpp"
#include "container/matrix.hpp"
#include "linear/factorization/cholesky.hpp"
#include "linear/factorization/lu.hpp"
#include "linear/matrix_properties.hpp"
#include "linear/sparse/sparse.hpp"
#include <gtest/gtest.h>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

/// Sparse builder that also records the dense equivalent, so tests never have to
/// keep two descriptions of the same matrix in step by hand.
class Builder {
  public:
    explicit Builder(num::idx n) : n_(n), dense_(n, n, 0.0) {}

    void add(num::idx row, num::idx col, double value) {
        rows_.push_back(row);
        cols_.push_back(col);
        values_.push_back(value);
        dense_(row, col) += value;
    }

    [[nodiscard]] num::spmat sparse() const {
        return num::spmat::from_triplets(n_, n_, rows_, cols_, values_);
    }
    [[nodiscard]] const num::mat &dense() const { return dense_; }

  private:
    num::idx n_;
    std::vector<num::idx> rows_;
    std::vector<num::idx> cols_;
    std::vector<double> values_;
    num::mat dense_;
};

/// Block-tridiagonal matrix with the given block sizes, diagonally dominant so
/// the unpivoted block elimination is well conditioned.
struct Problem {
    num::spmat sparse;
    num::mat dense;
    std::vector<num::idx> levels;
    num::idx size;
};

Problem make_problem(const std::vector<num::idx> &block_sizes, std::uint64_t seed,
                     bool symmetric = false, const std::vector<num::idx> *labels = nullptr) {
    num::idx n = 0;
    for (const num::idx size : block_sizes) {
        n += size;
    }
    Builder builder(n);
    std::vector<num::idx> levels(n, 0);
    std::vector<num::idx> first(block_sizes.size(), 0);
    num::idx running = 0;
    for (num::idx k = 0; k < block_sizes.size(); ++k) {
        first[k] = running;
        for (num::idx r = 0; r < block_sizes[k]; ++r) {
            levels[running + r] = labels != nullptr ? (*labels)[k] : k;
        }
        running += block_sizes[k];
    }

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> entry(-1.0, 1.0);

    for (num::idx k = 0; k < block_sizes.size(); ++k) {
        // Diagonal block, made strongly diagonally dominant.
        for (num::idx i = 0; i < block_sizes[k]; ++i) {
            for (num::idx j = 0; j < block_sizes[k]; ++j) {
                if (i == j) {
                    continue;
                }
                const double value = entry(rng);
                builder.add(first[k] + i, first[k] + j, value);
            }
            builder.add(first[k] + i, first[k] + i, 20.0);
        }
        if (k + 1 < block_sizes.size()) {
            for (num::idx i = 0; i < block_sizes[k]; ++i) {
                for (num::idx j = 0; j < block_sizes[k + 1]; ++j) {
                    const double up = entry(rng);
                    const double down = symmetric ? up : entry(rng);
                    builder.add(first[k] + i, first[k + 1] + j, up);
                    builder.add(first[k + 1] + j, first[k] + i, down);
                }
            }
        }
    }

    return Problem{builder.sparse(), builder.dense(), std::move(levels), n};
}

/// Symmetric positive-definite block-tridiagonal problem: build B then use B B^T + cI.
Problem make_spd_problem(const std::vector<num::idx> &block_sizes, std::uint64_t seed) {
    num::idx n = 0;
    for (const num::idx size : block_sizes) {
        n += size;
    }
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> entry(-0.4, 0.4);

    num::mat dense(n, n, 0.0);
    std::vector<num::idx> levels(n, 0);
    std::vector<num::idx> first(block_sizes.size(), 0);
    num::idx running = 0;
    for (num::idx k = 0; k < block_sizes.size(); ++k) {
        first[k] = running;
        for (num::idx r = 0; r < block_sizes[k]; ++r) {
            levels[running + r] = k;
        }
        running += block_sizes[k];
    }

    for (num::idx k = 0; k < block_sizes.size(); ++k) {
        for (num::idx i = 0; i < block_sizes[k]; ++i) {
            for (num::idx j = i; j < block_sizes[k]; ++j) {
                const double value = i == j ? 12.0 : entry(rng);
                dense(first[k] + i, first[k] + j) = value;
                dense(first[k] + j, first[k] + i) = value;
            }
        }
        if (k + 1 < block_sizes.size()) {
            for (num::idx i = 0; i < block_sizes[k]; ++i) {
                for (num::idx j = 0; j < block_sizes[k + 1]; ++j) {
                    const double value = entry(rng);
                    dense(first[k] + i, first[k + 1] + j) = value;
                    dense(first[k + 1] + j, first[k] + i) = value;
                }
            }
        }
    }

    Builder builder(n);
    for (num::idx i = 0; i < n; ++i) {
        for (num::idx j = 0; j < n; ++j) {
            if (dense(i, j) != 0.0) {
                builder.add(i, j, dense(i, j));
            }
        }
    }

    return Problem{builder.sparse(), builder.dense(), std::move(levels), n};
}

num::vec make_rhs(num::idx n, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> entry(-2.0, 2.0);
    num::vec b(n, 0.0);
    for (num::idx i = 0; i < n; ++i) {
        b[i] = entry(rng);
    }
    return b;
}

num::mat make_rhs_matrix(num::idx n, num::idx columns, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> entry(-2.0, 2.0);
    num::mat B(n, columns, 0.0);
    for (num::idx i = 0; i < n; ++i) {
        for (num::idx c = 0; c < columns; ++c) {
            B(i, c) = entry(rng);
        }
    }
    return B;
}

constexpr double tolerance = 1e-9;

// --- 1, 2: factorization equals dense LU -----------------------------------

TEST(BlockTridiagonal, SingleBlockMatchesDenseLU) {
    const auto problem = make_problem({6}, 11);
    const auto factor = num::factor_block_lu(problem.sparse, problem.levels);
    ASSERT_EQ(factor.blocks(), 1u);
    ASSERT_FALSE(factor.singular());

    const auto reference = num::lu(num::assume_square(problem.dense));
    const auto b = make_rhs(problem.size, 12);
    num::vec expected(problem.size, 0.0);
    num::vec actual(problem.size, 0.0);
    num::lu_solve(reference, b, expected);
    num::solve(factor, b, actual);
    for (num::idx i = 0; i < problem.size; ++i) {
        EXPECT_NEAR(actual[i], expected[i], tolerance) << "row " << i;
    }
}

TEST(BlockTridiagonal, MultipleBlocksMatchDenseLU) {
    const auto problem = make_problem({3, 4, 2, 5}, 21);
    const auto factor = num::factor_block_lu(problem.sparse, problem.levels);
    ASSERT_EQ(factor.blocks(), 4u);
    ASSERT_FALSE(factor.singular());

    const auto reference = num::lu(num::assume_square(problem.dense));
    const auto b = make_rhs(problem.size, 22);
    num::vec expected(problem.size, 0.0);
    num::vec actual(problem.size, 0.0);
    num::lu_solve(reference, b, expected);
    num::solve(factor, b, actual);
    for (num::idx i = 0; i < problem.size; ++i) {
        EXPECT_NEAR(actual[i], expected[i], tolerance) << "row " << i;
    }
}

// --- 3, 5: matrix right-hand sides ------------------------------------------

TEST(BlockTridiagonal, MatrixSolveMatchesDenseSolve) {
    const auto problem = make_problem({4, 3, 4}, 31);
    const auto factor = num::factor_block_lu(problem.sparse, problem.levels);
    const auto reference = num::lu(num::assume_square(problem.dense));

    const auto B = make_rhs_matrix(problem.size, 3, 32);
    num::mat expected;
    num::mat actual;
    num::lu_solve(reference, B, expected);
    num::solve(factor, B, actual);

    ASSERT_EQ(actual.rows(), problem.size);
    ASSERT_EQ(actual.cols(), 3u);
    for (num::idx i = 0; i < problem.size; ++i) {
        for (num::idx c = 0; c < 3; ++c) {
            EXPECT_NEAR(actual(i, c), expected(i, c), tolerance) << "(" << i << "," << c << ")";
        }
    }
}

TEST(BlockTridiagonal, ManyRightHandSidesAgreeColumnwiseWithSingleSolves) {
    const auto problem = make_problem({2, 5, 3}, 41);
    const auto factor = num::factor_block_lu(problem.sparse, problem.levels);

    const auto B = make_rhs_matrix(problem.size, 7, 42);
    num::mat actual;
    num::solve(factor, B, actual);
    ASSERT_EQ(actual.cols(), 7u);

    for (num::idx c = 0; c < 7; ++c) {
        num::vec column(problem.size, 0.0);
        for (num::idx i = 0; i < problem.size; ++i) {
            column[i] = B(i, c);
        }
        num::vec single(problem.size, 0.0);
        num::solve(factor, column, single);
        for (num::idx i = 0; i < problem.size; ++i) {
            EXPECT_NEAR(actual(i, c), single[i], tolerance) << "column " << c << " row " << i;
        }
    }
}

// --- 4: transpose solve ------------------------------------------------------

TEST(BlockTridiagonal, TransposeSolveMatchesDenseTransposeSolve) {
    const auto problem = make_problem({3, 5, 2}, 51);
    const auto factor = num::factor_block_lu(problem.sparse, problem.levels);
    const auto reference = num::lu(num::assume_square(problem.dense));

    const auto b = make_rhs(problem.size, 52);
    num::vec expected(problem.size, 0.0);
    num::vec actual(problem.size, 0.0);
    num::lu_solve_transpose(reference, b, expected);
    num::solve_transpose(factor, b, actual);
    for (num::idx i = 0; i < problem.size; ++i) {
        EXPECT_NEAR(actual[i], expected[i], tolerance) << "row " << i;
    }

    const auto B = make_rhs_matrix(problem.size, 4, 53);
    num::mat expected_matrix;
    num::mat actual_matrix;
    num::lu_solve_transpose(reference, B, expected_matrix);
    num::solve_transpose(factor, B, actual_matrix);
    for (num::idx i = 0; i < problem.size; ++i) {
        for (num::idx c = 0; c < 4; ++c) {
            EXPECT_NEAR(actual_matrix(i, c), expected_matrix(i, c), tolerance)
                << "(" << i << "," << c << ")";
        }
    }
}

TEST(BlockTridiagonal, TransposeSolveReusesTheFactorsWithoutRefactorizing) {
    // A^T (A^-1 b) must return b: the two directions share one set of factors.
    const auto problem = make_problem({4, 4, 4}, 61);
    const auto factor = num::factor_block_lu(problem.sparse, problem.levels);
    const auto b = make_rhs(problem.size, 62);

    num::vec y(problem.size, 0.0);
    num::solve_transpose(factor, b, y); // y = A^-T b
    num::vec back(problem.size, 0.0);
    // A^T y = b, so multiplying back through the dense transpose recovers b.
    for (num::idx i = 0; i < problem.size; ++i) {
        double sum = 0.0;
        for (num::idx j = 0; j < problem.size; ++j) {
            sum += problem.dense(j, i) * y[j];
        }
        back[i] = sum;
    }
    for (num::idx i = 0; i < problem.size; ++i) {
        EXPECT_NEAR(back[i], b[i], tolerance) << "row " << i;
    }
}

// --- 6: Cholesky -------------------------------------------------------------

TEST(BlockTridiagonal, CholeskySolveMatchesDenseCholesky) {
    const auto problem = make_spd_problem({3, 4, 3}, 71);
    const auto factor = num::factor_block_cholesky(problem.sparse, problem.levels);
    ASSERT_EQ(factor.blocks(), 3u);
    ASSERT_FALSE(factor.failed());

    const auto reference = num::cholesky(num::assume_spd(problem.dense));
    ASSERT_TRUE(reference.success);

    const auto b = make_rhs(problem.size, 72);
    num::vec expected(problem.size, 0.0);
    num::vec actual(problem.size, 0.0);
    num::cholesky_solve(reference, b, expected);
    num::solve(factor, b, actual);
    for (num::idx i = 0; i < problem.size; ++i) {
        EXPECT_NEAR(actual[i], expected[i], tolerance) << "row " << i;
    }

    const auto B = make_rhs_matrix(problem.size, 3, 73);
    num::mat expected_matrix;
    num::mat actual_matrix;
    num::cholesky_solve(reference, B, expected_matrix);
    num::solve(factor, B, actual_matrix);
    for (num::idx i = 0; i < problem.size; ++i) {
        for (num::idx c = 0; c < 3; ++c) {
            EXPECT_NEAR(actual_matrix(i, c), expected_matrix(i, c), tolerance)
                << "(" << i << "," << c << ")";
        }
    }
}

// --- 7: structural rejection -------------------------------------------------

TEST(BlockTridiagonal, RejectsNonBlockTridiagonalSparsity) {
    Builder builder(6);
    for (num::idx i = 0; i < 6; ++i) {
        builder.add(i, i, 5.0);
    }
    builder.add(0, 5, 1.0); // couples block 0 to block 2
    const std::vector<num::idx> levels{0, 0, 1, 1, 2, 2};
    EXPECT_THROW((void)num::factor_block_lu(builder.sparse(), levels), std::invalid_argument);
}

TEST(BlockTridiagonal, RejectsMismatchedShapeOrLevelCount) {
    Builder builder(4);
    for (num::idx i = 0; i < 4; ++i) {
        builder.add(i, i, 3.0);
    }
    const std::vector<num::idx> too_few{0, 0, 1};
    EXPECT_THROW((void)num::factor_block_lu(builder.sparse(), too_few), std::invalid_argument);

    const std::vector<num::idx> rows{0, 1};
    const std::vector<num::idx> cols{0, 1};
    const std::vector<double> values{1.0, 1.0};
    const auto rectangular = num::spmat::from_triplets(2, 3, rows, cols, values);
    const std::vector<num::idx> levels{0, 1};
    EXPECT_THROW((void)num::factor_block_lu(rectangular, levels), std::invalid_argument);
}

TEST(BlockTridiagonal, AcceptsAnExplicitZeroOutsideTheBand) {
    // A stored zero contributes nothing, so rejecting it would make a padded
    // pattern unusable for no numerical reason.
    Builder builder(6);
    for (num::idx i = 0; i < 6; ++i) {
        builder.add(i, i, 5.0);
    }
    builder.add(0, 5, 0.0);
    const std::vector<num::idx> levels{0, 0, 1, 1, 2, 2};
    EXPECT_NO_THROW((void)num::factor_block_lu(builder.sparse(), levels));
}

// --- 8: unequal block sizes --------------------------------------------------

TEST(BlockTridiagonal, HandlesStronglyUnequalBlockSizes) {
    const auto problem = make_problem({1, 7, 2, 1, 6}, 81);
    const auto factor = num::factor_block_lu(problem.sparse, problem.levels);
    ASSERT_EQ(factor.blocks(), 5u);
    EXPECT_EQ(factor.block_size(0), 1u);
    EXPECT_EQ(factor.block_size(1), 7u);
    EXPECT_EQ(factor.block_size(3), 1u);

    const auto reference = num::lu(num::assume_square(problem.dense));
    const auto b = make_rhs(problem.size, 82);
    num::vec expected(problem.size, 0.0);
    num::vec actual(problem.size, 0.0);
    num::lu_solve(reference, b, expected);
    num::solve(factor, b, actual);
    for (num::idx i = 0; i < problem.size; ++i) {
        EXPECT_NEAR(actual[i], expected[i], tolerance) << "row " << i;
    }
}

// --- 9: arbitrary level labels -----------------------------------------------

TEST(BlockTridiagonal, CompressesArbitraryNonContiguousLevelLabels) {
    // Labels 40, 17, 900 must compress to blocks 0, 1, 2 in sorted order and give
    // the same answer as the equivalent 0, 1, 2 labelling.
    const std::vector<num::idx> sizes{3, 2, 4};
    const std::vector<num::idx> odd_labels{17, 40, 900};
    const auto plain = make_problem(sizes, 91);
    const auto relabelled = make_problem(sizes, 91, false, &odd_labels);

    const auto factor_plain = num::factor_block_lu(plain.sparse, plain.levels);
    const auto factor_odd = num::factor_block_lu(relabelled.sparse, relabelled.levels);
    ASSERT_EQ(factor_odd.blocks(), 3u);
    EXPECT_EQ(factor_odd.offsets, factor_plain.offsets);

    const auto b = make_rhs(plain.size, 92);
    num::vec from_plain(plain.size, 0.0);
    num::vec from_odd(plain.size, 0.0);
    num::solve(factor_plain, b, from_plain);
    num::solve(factor_odd, b, from_odd);
    for (num::idx i = 0; i < plain.size; ++i) {
        EXPECT_NEAR(from_odd[i], from_plain[i], tolerance) << "row " << i;
    }
}

TEST(BlockTridiagonal, HandlesLevelsGivenOutOfOrder) {
    // Rows interleaved across levels: the permutation, not the row index, decides
    // block membership. A factorization that assumed sorted input would fail here.
    Builder builder(6);
    const std::vector<num::idx> levels{1, 0, 1, 0, 2, 2};
    // block 0 = rows {1,3}, block 1 = rows {0,2}, block 2 = rows {4,5}
    for (num::idx i = 0; i < 6; ++i) {
        builder.add(i, i, 10.0);
    }
    builder.add(1, 0, 1.5); // block 0 -> block 1
    builder.add(0, 1, 0.5);
    builder.add(3, 2, -1.0);
    builder.add(2, 3, 2.0);
    builder.add(2, 4, 1.0); // block 1 -> block 2
    builder.add(4, 2, -0.5);

    const auto sparse = builder.sparse();
    const auto factor = num::factor_block_lu(sparse, levels);
    ASSERT_EQ(factor.blocks(), 3u);
    EXPECT_EQ(factor.order[0], 1u);
    EXPECT_EQ(factor.order[1], 3u);

    const auto reference = num::lu(num::assume_square(builder.dense()));
    const auto b = make_rhs(6, 101);
    num::vec expected(6, 0.0);
    num::vec actual(6, 0.0);
    num::lu_solve(reference, b, expected);
    num::solve(factor, b, actual);
    for (num::idx i = 0; i < 6; ++i) {
        EXPECT_NEAR(actual[i], expected[i], tolerance) << "row " << i;
    }
}

// --- 10: low-rank update preparation ----------------------------------------

TEST(BlockTridiagonal, AppliedToATallBlockMatchesDenseSolveForWoodbury) {
    // What a block-Woodbury layer needs: A0^-1 U and A0^-T V from the stored
    // factors, then the small K = I + V^T A0^-1 U. Both applications must agree
    // with the dense solve entrywise, or K is formed from the wrong operator.
    const auto problem = make_problem({4, 3, 5}, 111);
    const auto factor = num::factor_block_lu(problem.sparse, problem.levels);
    const auto reference = num::lu(num::assume_square(problem.dense));

    const num::idx rank = 3;
    const auto U = make_rhs_matrix(problem.size, rank, 112);
    const auto V = make_rhs_matrix(problem.size, rank, 113);

    num::mat applied;
    num::mat expected;
    num::solve(factor, U, applied);
    num::lu_solve(reference, U, expected);
    for (num::idx i = 0; i < problem.size; ++i) {
        for (num::idx c = 0; c < rank; ++c) {
            EXPECT_NEAR(applied(i, c), expected(i, c), tolerance)
                << "A^-1 U (" << i << "," << c << ")";
        }
    }

    num::mat applied_transpose;
    num::mat expected_transpose;
    num::solve_transpose(factor, V, applied_transpose);
    num::lu_solve_transpose(reference, V, expected_transpose);
    for (num::idx i = 0; i < problem.size; ++i) {
        for (num::idx c = 0; c < rank; ++c) {
            EXPECT_NEAR(applied_transpose(i, c), expected_transpose(i, c), tolerance)
                << "A^-T V (" << i << "," << c << ")";
        }
    }

    // K = I + V^T A^-1 U, formed from the block factor, matches the dense one.
    for (num::idx a = 0; a < rank; ++a) {
        for (num::idx c = 0; c < rank; ++c) {
            double from_block = a == c ? 1.0 : 0.0;
            double from_dense = from_block;
            for (num::idx i = 0; i < problem.size; ++i) {
                from_block += V(i, a) * applied(i, c);
                from_dense += V(i, a) * expected(i, c);
            }
            EXPECT_NEAR(from_block, from_dense, tolerance) << "K(" << a << "," << c << ")";
        }
    }
}

TEST(BlockTridiagonal, ExposesTheLayoutALowRankLayerNeeds) {
    const auto problem = make_problem({2, 3, 4}, 121);
    const auto factor = num::factor_block_lu(problem.sparse, problem.levels);
    EXPECT_EQ(factor.size, problem.size);
    EXPECT_EQ(factor.offsets.size(), 4u);
    EXPECT_EQ(factor.order.size(), problem.size);
    EXPECT_EQ(factor.diagonal.size(), 3u);
    EXPECT_EQ(factor.upper.size(), 2u);
    EXPECT_EQ(factor.lower.size(), 2u);
    EXPECT_EQ(factor.upper[0].rows(), 2u);
    EXPECT_EQ(factor.upper[0].cols(), 3u);
    EXPECT_EQ(factor.lower[0].rows(), 3u);
    EXPECT_EQ(factor.lower[0].cols(), 2u);
}

} // namespace
