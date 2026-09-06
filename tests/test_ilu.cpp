/// @file tests/test_ilu.cpp
/// @brief ILU(0) kernels, the preconditioner over them, and right-preconditioned GMRES.
///
/// The sharpest check available is exactness: for a tridiagonal matrix, Gaussian
/// elimination creates no entries outside the existing pattern, so ILU(0)
/// discards nothing and the factorization is a *complete* LU. Preconditioned
/// GMRES must then converge in a single iteration. Anything wrong with the
/// factorization, the triangular solves, or the preconditioned recurrence breaks
/// that, and breaks it loudly.

#include "linear/solvers/ilu.hpp"
#include "kernel/kernel.hpp"
#include "linear/solvers/math_gmres.hpp"
#include "linear/sparse/sparse.hpp"
#include "linear/sparse/sparse_op.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

namespace {

/// tridiagonal matrix with a strong diagonal, optionally nonsymmetric.
num::spmat tridiagonal(num::idx n, double lower = -1.0, double upper = -2.0,
                              double diag = 6.0) {
    std::vector<num::idx> rows;
    std::vector<num::idx> cols;
    std::vector<double> values;
    for (num::idx i = 0; i < n; ++i) {
        if (i > 0) {
            rows.push_back(i);
            cols.push_back(i - 1);
            values.push_back(lower);
        }
        rows.push_back(i);
        cols.push_back(i);
        values.push_back(diag);
        if (i + 1 < n) {
            rows.push_back(i);
            cols.push_back(i + 1);
            values.push_back(upper);
        }
    }
    return num::spmat::from_triplets(n, n, rows, cols, values);
}

// --- kernel tier ------------------------------------------------------------

TEST(ILUKernel, FindsDiagonalPositionsAndReportsAMissingOne) {
    const auto A = tridiagonal(5);
    std::vector<num::idx> diagonal(5, 0);
    ASSERT_TRUE(num::kernel::csr_diagonal_positions(diagonal.data(), A.row_ptr(), A.col_idx(),
                                                         num::idx{5}));
    for (num::idx i = 0; i < 5; ++i) {
        EXPECT_EQ(A.col_idx()[diagonal[i]], i) << "row " << i;
    }

    // A matrix whose second row has no stored diagonal.
    const std::vector<num::idx> rows{0, 1};
    const std::vector<num::idx> cols{0, 0};
    const std::vector<double> values{1.0, 1.0};
    const auto gap = num::spmat::from_triplets(2, 2, rows, cols, values);
    std::vector<num::idx> d2(2, 0);
    EXPECT_FALSE(num::kernel::csr_diagonal_positions(d2.data(), gap.row_ptr(), gap.col_idx(),
                                                          num::idx{2}));
}

TEST(ILUKernel, FactorizationIsExactWhenNoFillInIsDiscarded) {
    // tridiagonal: elimination never leaves the pattern, so LU is complete and
    // solving with it must reproduce the exact solution of A x = b.
    const num::idx n = 40;
    const auto A = tridiagonal(n);
    const num::ilu0_preconditioner M(A);

    num::vec x_exact(n);
    for (num::idx i = 0; i < n; ++i) {
        x_exact[i] = std::sin(0.4 * static_cast<double>(i)) + 1.5;
    }
    num::vec b(n, 0.0);
    num::sparse_matvec(A, x_exact, b);

    num::vec x(n, 0.0);
    M.apply(b, x);
    for (num::idx i = 0; i < n; ++i) {
        EXPECT_NEAR(x[i], x_exact[i], 1e-10) << "element " << i;
    }
}

TEST(ILUKernel, ReportsAZeroPivotRatherThanProducingGarbage) {
    // A singular leading block drives a pivot to zero during elimination.
    const std::vector<num::idx> rows{0, 0, 1, 1};
    const std::vector<num::idx> cols{0, 1, 0, 1};
    const std::vector<double> values{1.0, 1.0, 1.0, 1.0}; // second row eliminates to zero
    const auto A = num::spmat::from_triplets(2, 2, rows, cols, values);
    EXPECT_THROW((void)num::make_ilu0_preconditioner(A), std::runtime_error);
}

// --- preconditioner ---------------------------------------------------------

TEST(ILU0, RejectsAMatrixItCannotFactor) {
    const std::vector<num::idx> rows{0, 1};
    const std::vector<num::idx> cols{0, 1};
    const std::vector<double> values{1.0, 1.0};
    const auto rectangular = num::spmat::from_triplets(2, 3, rows, cols, values);
    EXPECT_THROW((void)num::make_ilu0_preconditioner(rectangular), std::invalid_argument);

    // Missing diagonal entry.
    const std::vector<num::idx> r2{0, 1};
    const std::vector<num::idx> c2{1, 0};
    const std::vector<double> v2{1.0, 1.0};
    const auto no_diagonal = num::spmat::from_triplets(2, 2, r2, c2, v2);
    EXPECT_THROW((void)num::make_ilu0_preconditioner(no_diagonal), std::invalid_argument);
}

TEST(ILU0, PreservesTheSparsityPatternExactly) {
    const auto A = tridiagonal(30);
    const num::ilu0_preconditioner M(A);
    EXPECT_EQ(M.nnz(), A.nnz()) << "zero fill-in means the factors cost what the matrix costs";
    EXPECT_EQ(M.rows(), A.n_rows());
}

TEST(ILU0, ApplicationIsRepeatable) {
    const auto A = tridiagonal(50);
    const num::ilu0_preconditioner M(A);
    num::vec r(50, 1.25);
    num::vec first(50, 0.0);
    num::vec second(50, 0.0);
    M.apply(r, first);
    M.apply(r, second);
    for (num::idx i = 0; i < 50; ++i) {
        EXPECT_DOUBLE_EQ(first[i], second[i]) << "element " << i;
    }
}

// --- right-preconditioned GMRES --------------------------------------------

TEST(PreconditionedGMRES, ConvergesInOneStepWhenTheFactorizationIsExact) {
    const num::idx n = 60;
    const auto A = tridiagonal(n);
    const num::operators::sparse_op op(A);
    const num::ilu0_preconditioner M(A);

    num::vec b(n, 1.0);
    num::vec x(n, 0.0);
    const auto result =
        num::gmres(op, M, b, x, num::gmres_options{.tolerance = 1e-10, .max_iterations = 100});

    EXPECT_TRUE(result.converged);
    EXPECT_LE(result.iterations, 2u) << "an exact preconditioner leaves nothing for Krylov to do";
}

TEST(PreconditionedGMRES, ReportsTheTrueResidualNotThePreconditionedOne) {
    // This is the reason for right rather than left preconditioning: the number
    // the caller reads must be ||b - Ax||, whatever M does to the scaling.
    const num::idx n = 80;
    const auto A = tridiagonal(n, -1.0, -3.0, 9.0);
    const num::operators::sparse_op op(A);
    const num::ilu0_preconditioner M(A);

    num::vec b(n, 2.0);
    num::vec x(n, 0.0);
    const auto result =
        num::gmres(op, M, b, x, num::gmres_options{.tolerance = 1e-9, .max_iterations = 200});
    ASSERT_TRUE(result.converged);

    num::vec residual(n, 0.0);
    num::sparse_matvec(A, x, residual);
    num::math::linear_combination(num::real(1), b, num::real(-1), residual);
    EXPECT_NEAR(result.residual, num::math::norm(residual), 1e-8 * std::max(1.0, result.residual));
}

TEST(PreconditionedGMRES, CutsIterationsOnANonsymmetricSystem) {
    // Upwind convection-diffusion: nonsymmetric, and the case ILU(0) is for.
    const num::idx side = 30;
    const num::idx n = side * side;
    const double nu = 0.01;
    const double h = 1.0 / static_cast<double>(side + 1);
    std::vector<num::idx> rows;
    std::vector<num::idx> cols;
    std::vector<double> values;
    auto push = [&](num::idx i, num::idx j, double a) {
        rows.push_back(i);
        cols.push_back(j);
        values.push_back(a);
    };
    for (num::idx i = 0; i < side; ++i) {
        for (num::idx j = 0; j < side; ++j) {
            const num::idx k = (i * side) + j;
            if (i > 0) {
                push(k, k - side, (-nu / (h * h)) - (1.0 / h));
            }
            if (i + 1 < side) {
                push(k, k + side, -nu / (h * h));
            }
            if (j > 0) {
                push(k, k - 1, (-nu / (h * h)) - (1.0 / h));
            }
            if (j + 1 < side) {
                push(k, k + 1, -nu / (h * h));
            }
            push(k, k, (4.0 * nu / (h * h)) + (2.0 / h));
        }
    }
    const auto A = num::spmat::from_triplets(n, n, rows, cols, values);
    const num::operators::sparse_op op(A);

    num::vec b(n, 1.0);
    num::vec x_plain(n, 0.0);
    const auto plain = num::gmres(
        op, b, x_plain, num::gmres_options{.tolerance = 1e-8, .max_iterations = 5000, .restart = 50});
    ASSERT_TRUE(plain.converged);

    const num::ilu0_preconditioner M(A);
    num::vec x_prec(n, 0.0);
    const auto preconditioned =
        num::gmres(op, M, b, x_prec,
                   num::gmres_options{.tolerance = 1e-8, .max_iterations = 5000, .restart = 50});
    ASSERT_TRUE(preconditioned.converged);

    EXPECT_LT(preconditioned.iterations * 4, plain.iterations)
        << "expected a large reduction, got " << plain.iterations << " -> "
        << preconditioned.iterations;

    for (num::idx i = 0; i < n; i += 137) {
        EXPECT_NEAR(x_prec[i], x_plain[i], 1e-5 * std::max(1.0, std::abs(x_plain[i])));
    }
}

} // namespace
