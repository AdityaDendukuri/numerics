/// @file tests/test_lu_no_pivot.cpp
/// @brief Pivot-free LU against dense `num::lu` on a diagonally dominant matrix.
///
/// Diagonal dominance is exactly the structural guarantee `num::lu(A, num::no_pivot)`
/// requires: it rules out a zero or tiny pivot, so skipping the row search and
/// swap `num::lu` performs is safe. Every test below compares against `num::lu`
/// on the same matrix, so a wrong forward/back substitution shows up as soon as
/// the two solutions diverge.

#include "linear/factorization/factor.hpp"
#include "linear/factorization/lu.hpp"
#include "linear/matrix_properties.hpp"
#include <gtest/gtest.h>

using namespace num;

namespace {

/// Strictly diagonally dominant by rows, so `lu(A, no_pivot)` never reports
/// `singular` and the elimination never meets a zero pivot.
mat<real> make_diagonally_dominant(idx n) {
    mat<real> A(n, n, 0.0);
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
    mat<real> A = make_diagonally_dominant(5);
    const auto factor = lu(A, no_pivot);
    EXPECT_FALSE(factor.singular);
}

TEST(LUNoPivot, SolveMatchesPivotedLU) {
    const idx n = 6;
    mat<real> A = make_diagonally_dominant(n);
    vec<real> b(n);
    for (idx i = 0; i < n; ++i) {
        b[i] = static_cast<real>(i + 1);
    }

    const auto factor = lu(A, no_pivot);
    ASSERT_FALSE(factor.singular);
    vec<real> x(n);
    solve(factor, b, x);

    const auto reference = lu(A);
    vec<real> x_ref(n);
    solve(reference, b, x_ref);

    for (idx i = 0; i < n; ++i) {
        EXPECT_NEAR(x[i], x_ref[i], 1e-9);
    }
}

TEST(LUNoPivot, SolveMultipleRHSMatchesPivotedLU) {
    const idx n = 5;
    mat<real> A = make_diagonally_dominant(n);
    mat<real> B(n, 2, 0.0);
    for (idx i = 0; i < n; ++i) {
        B(i, 0) = static_cast<real>(i + 1);
        B(i, 1) = static_cast<real>(n - i);
    }

    const auto factor = lu(A, no_pivot);
    mat<real> X;
    solve(factor, B, X);

    const auto reference = lu(A);
    mat<real> X_ref;
    solve(reference, B, X_ref);

    for (idx i = 0; i < n; ++i) {
        for (idx j = 0; j < B.cols(); ++j) {
            EXPECT_NEAR(X(i, j), X_ref(i, j), 1e-9);
        }
    }
}

TEST(LUNoPivot, SolveTransposeMatchesPivotedLU) {
    const idx n = 5;
    mat<real> A = make_diagonally_dominant(n);
    vec<real> b(n);
    for (idx i = 0; i < n; ++i) {
        b[i] = static_cast<real>(2 * i + 1);
    }

    const auto factor = lu(A, no_pivot);
    vec<real> x(n);
    solve_transpose(factor, b, x);

    const auto reference = lu(A);
    vec<real> x_ref(n);
    solve_transpose(reference, b, x_ref);

    for (idx i = 0; i < n; ++i) {
        EXPECT_NEAR(x[i], x_ref[i], 1e-9);
    }
}

TEST(LUNoPivot, ReportsSingularOnZeroPivot) {
    // No diagonal dominance at all: the (0,0) pivot is exactly zero.
    mat<real> A(3, 3, 0.0);
    A(0, 1) = 1.0;
    A(1, 0) = 1.0;
    A(1, 1) = 2.0;
    A(2, 2) = 3.0;

    const auto factor = lu(A, no_pivot);
    EXPECT_TRUE(factor.singular);
}

namespace {

/// A matrix whose first column forces partial pivoting to swap rows.
mat<real> make_needs_pivoting(idx n) {
    mat<real> A = make_diagonally_dominant(n);
    for (idx i = 0; i < n; ++i) {
        A(i, 0) = static_cast<real>(i + 1);
    }
    return A;
}

mat<real> product(const mat<real> &L, const mat<real> &U) {
    const idx n = L.rows();
    mat<real> P(n, n, 0.0);
    for (idx i = 0; i < n; ++i)
        for (idx k = 0; k < n; ++k)
            for (idx j = 0; j < n; ++j)
                P(i, j) += L(i, k) * U(k, j);
    return P;
}

} // namespace

TEST(LUFactors, LowerTimesUpperRebuildsPermutedMatrix) {
    const idx n = 6;
    const mat<real> A = make_needs_pivoting(n);
    const lu_result<real> f = lu(A);
    ASSERT_EQ(f.swaps.size(), n);

    mat<real> PA = A;
    for (idx k = 0; k < n; ++k) {
        for (idx j = 0; j < n; ++j) {
            std::swap(PA(k, j), PA(f.swaps[k], j));
        }
    }
    const mat<real> LU = product(lower(f), upper(f));
    for (idx i = 0; i < n; ++i) {
        EXPECT_EQ(lower(f)(i, i), 1.0);
        for (idx j = 0; j < n; ++j) {
            EXPECT_NEAR(LU(i, j), PA(i, j), 1e-12);
        }
    }
}

TEST(LUFactors, NoPivotHasNoSwapsAndRebuildsMatrix) {
    const idx n = 5;
    const mat<real> A = make_diagonally_dominant(n);
    const lu_result<real> f = lu(A, no_pivot);
    EXPECT_TRUE(f.swaps.empty());
    const mat<real> LU = product(lower(f), upper(f));
    for (idx i = 0; i < n; ++i) {
        for (idx j = 0; j < n; ++j) {
            EXPECT_NEAR(LU(i, j), A(i, j), 1e-12);
        }
    }
}

TEST(LUFactors, DetAndInverseAgreeWithAndWithoutPivoting) {
    const idx n = 5;
    const mat<real> A = make_diagonally_dominant(n);
    const lu_result<real> pivoted = lu(make_needs_pivoting(n));
    const lu_result<real> plain = lu(A, no_pivot);
    const lu_result<real> reference = lu(A);

    EXPECT_NEAR(det(plain), det(reference), 1e-9 * std::abs(det(reference)));
    const mat<real> inv = inverse(plain);
    const mat<real> inv_ref = inverse(reference);
    for (idx i = 0; i < n; ++i) {
        for (idx j = 0; j < n; ++j) {
            EXPECT_NEAR(inv(i, j), inv_ref(i, j), 1e-12);
        }
    }
    // The pivoted determinant's sign comes from the swaps: check it against L U.
    real diagonal = 1.0;
    for (idx i = 0; i < n; ++i) {
        diagonal *= pivoted.LU(i, i);
    }
    EXPECT_NEAR(std::abs(det(pivoted)), std::abs(diagonal), 1e-12 * std::abs(diagonal));
}

TEST(LUFactors, GenericSolveAndTransposeViewMatchExplicitCalls) {
    const idx n = 6;
    const lu_result<real> f = lu(make_needs_pivoting(n));
    vec<real> b(n);
    for (idx i = 0; i < n; ++i) {
        b[i] = static_cast<real>(i) - 2.5;
    }

    vec<real> x_ref, xt_ref;
    solve(f, b, x_ref);
    solve_transpose(f, b, xt_ref);

    const vec<real> x = solve(f, b);
    const vec<real> xt = solve(transpose(f), b);
    vec<real> aliased = b;
    solve(f, aliased, aliased);
    for (idx i = 0; i < n; ++i) {
        EXPECT_EQ(x[i], x_ref[i]);
        EXPECT_EQ(xt[i], xt_ref[i]);
        EXPECT_EQ(aliased[i], x_ref[i]);
    }
}

TEST(LUFactors, TransposeSolveVectorMatchesMatrixPath) {
    const idx n = 7;
    const lu_result<real> f = lu(make_needs_pivoting(n));
    mat<real> B(n, 1, 0.0);
    vec<real> b(n);
    for (idx i = 0; i < n; ++i) {
        b[i] = B(i, 0) = static_cast<real>(3 * i + 1);
    }
    vec<real> x;
    mat<real> X;
    solve_transpose(f, b, x);
    solve_transpose(f, B, X);
    for (idx i = 0; i < n; ++i) {
        EXPECT_NEAR(x[i], X(i, 0), 1e-12);
    }
}
