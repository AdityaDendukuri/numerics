/// @file tests/test_lu_no_pivot.cpp
/// @brief Pivot-free LU against dense `num::lu` on a diagonally dominant matrix.
///
/// Diagonal dominance is exactly the structural guarantee `num::factor_no_pivot`
/// requires: it rules out a zero or tiny pivot, so skipping the row search and
/// swap `num::lu` performs is safe. Every test below compares against `num::lu`
/// on the same matrix, so a wrong forward/back substitution shows up as soon as
/// the two solutions diverge.

#include "linear/factorization/lu.hpp"
#include "linear/factorization/lu_no_pivot.hpp"
#include "linear/matrix_properties.hpp"
#include <gtest/gtest.h>

using namespace num;

namespace {

/// Strictly diagonally dominant by rows, so `factor_no_pivot` never reports
/// `singular` and the elimination never meets a zero pivot.
mat make_diagonally_dominant(idx n) {
    mat A(n, n, 0.0);
    for (idx i = 0; i < n; ++i) {
        real row_sum = 0.0;
        for (idx j = 0; j < n; ++j) {
            if (j == i) {
                continue;
            }
            A(i, j) = static_cast<real>(1) / static_cast<real>(1 + std::abs(static_cast<long>(i) -
                                                                            static_cast<long>(j)));
            row_sum += std::abs(A(i, j));
        }
        A(i, i) = row_sum + static_cast<real>(n);
    }
    return A;
}

} // namespace

TEST(LUNoPivot, NotSingularOnDiagonallyDominantMatrix) {
    mat A = make_diagonally_dominant(5);
    const auto factor = factor_no_pivot(assume_square(A));
    EXPECT_FALSE(factor.singular);
}

TEST(LUNoPivot, SolveMatchesPivotedLU) {
    const idx n = 6;
    mat A = make_diagonally_dominant(n);
    vec b(n);
    for (idx i = 0; i < n; ++i) {
        b[i] = static_cast<real>(i + 1);
    }

    const auto factor = factor_no_pivot(assume_square(A));
    ASSERT_FALSE(factor.singular);
    vec x(n);
    solve(factor, b, x);

    const auto reference = lu(assume_square(A));
    vec x_ref(n);
    lu_solve(reference, b, x_ref);

    for (idx i = 0; i < n; ++i) {
        EXPECT_NEAR(x[i], x_ref[i], 1e-9);
    }
}

TEST(LUNoPivot, SolveMultipleRHSMatchesPivotedLU) {
    const idx n = 5;
    mat A = make_diagonally_dominant(n);
    mat B(n, 2, 0.0);
    for (idx i = 0; i < n; ++i) {
        B(i, 0) = static_cast<real>(i + 1);
        B(i, 1) = static_cast<real>(n - i);
    }

    const auto factor = factor_no_pivot(assume_square(A));
    mat X;
    solve(factor, B, X);

    const auto reference = lu(assume_square(A));
    mat X_ref;
    lu_solve(reference, B, X_ref);

    for (idx i = 0; i < n; ++i) {
        for (idx j = 0; j < B.cols(); ++j) {
            EXPECT_NEAR(X(i, j), X_ref(i, j), 1e-9);
        }
    }
}

TEST(LUNoPivot, SolveTransposeMatchesPivotedLU) {
    const idx n = 5;
    mat A = make_diagonally_dominant(n);
    vec b(n);
    for (idx i = 0; i < n; ++i) {
        b[i] = static_cast<real>(2 * i + 1);
    }

    const auto factor = factor_no_pivot(assume_square(A));
    vec x(n);
    solve_transpose(factor, b, x);

    const auto reference = lu(assume_square(A));
    vec x_ref(n);
    lu_solve_transpose(reference, b, x_ref);

    for (idx i = 0; i < n; ++i) {
        EXPECT_NEAR(x[i], x_ref[i], 1e-9);
    }
}

TEST(LUNoPivot, ReportsSingularOnZeroPivot) {
    // No diagonal dominance at all: the (0,0) pivot is exactly zero.
    mat A(3, 3, 0.0);
    A(0, 1) = 1.0;
    A(1, 0) = 1.0;
    A(1, 1) = 2.0;
    A(2, 2) = 3.0;

    const auto factor = factor_no_pivot(assume_square(A));
    EXPECT_TRUE(factor.singular);
}
