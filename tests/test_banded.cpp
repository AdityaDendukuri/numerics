/// @file test_banded.cpp
/// @brief Tests for banded matrix solver

#include <gtest/gtest.h>
#include "linalg/banded/banded.hpp"
#include <cmath>
#include <cstring>
#include <random>

using namespace num;

// BandedMatrix Construction and Access Tests

TEST(BandedMatrix, ConstructBasic) {
    BandedMatrix A(10, 2, 3);  // 10x10 with 2 lower, 3 upper diagonals

    EXPECT_EQ(A.size(), 10);
    EXPECT_EQ(A.rows(), 10);
    EXPECT_EQ(A.cols(), 10);
    EXPECT_EQ(A.kl(), 2);
    EXPECT_EQ(A.ku(), 3);
    EXPECT_EQ(A.bandwidth(), 6);  // kl + ku + 1
    EXPECT_EQ(A.ldab(), 8);       // 2*kl + ku + 1
}

TEST(BandedMatrix, ConstructWithValue) {
    BandedMatrix A(5, 1, 1, 2.0);

    // Check that band elements are initialized
    for (idx j = 0; j < 5; ++j) {
        for (idx i = (j > 0 ? j - 1 : 0); i <= std::min(j + 1, idx(4)); ++i) {
            EXPECT_EQ(A(i, j), 2.0);
        }
    }
}

TEST(BandedMatrix, ElementAccess) {
    BandedMatrix A(5, 1, 2, 0.0);  // Tridiagonal plus one extra upper

    // Set diagonal
    for (idx i = 0; i < 5; ++i) {
        A(i, i) = static_cast<real>(i + 1) * 10.0;
    }

    // Set sub-diagonal
    for (idx i = 1; i < 5; ++i) {
        A(i, i - 1) = -1.0;
    }

    // Set super-diagonals
    for (idx i = 0; i < 4; ++i) {
        A(i, i + 1) = 2.0;
    }
    for (idx i = 0; i < 3; ++i) {
        A(i, i + 2) = 0.5;
    }

    // Verify values
    EXPECT_EQ(A(0, 0), 10.0);
    EXPECT_EQ(A(2, 2), 30.0);
    EXPECT_EQ(A(1, 0), -1.0);
    EXPECT_EQ(A(0, 1), 2.0);
    EXPECT_EQ(A(0, 2), 0.5);
}

TEST(BandedMatrix, InBandCheck) {
    BandedMatrix A(5, 1, 2, 0.0);

    // Diagonal is in band
    EXPECT_TRUE(A.in_band(2, 2));

    // Lower diagonal is in band
    EXPECT_TRUE(A.in_band(3, 2));

    // Upper diagonals in band
    EXPECT_TRUE(A.in_band(2, 3));
    EXPECT_TRUE(A.in_band(2, 4));

    // Outside band
    EXPECT_FALSE(A.in_band(0, 3));  // Too far above
    EXPECT_FALSE(A.in_band(4, 0));  // Too far below
}

TEST(BandedMatrix, CopyConstruct) {
    BandedMatrix A(4, 1, 1, 0.0);
    A(0, 0) = 2.0;
    A(0, 1) = -1.0;
    A(1, 0) = -1.0;
    A(1, 1) = 2.0;

    BandedMatrix B(A);

    EXPECT_EQ(B.size(), 4);
    EXPECT_EQ(B.kl(), 1);
    EXPECT_EQ(B.ku(), 1);
    EXPECT_EQ(B(0, 0), 2.0);
    EXPECT_EQ(B(0, 1), -1.0);

    // Ensure deep copy
    A(0, 0) = 999.0;
    EXPECT_EQ(B(0, 0), 2.0);
}

TEST(BandedMatrix, MoveConstruct) {
    BandedMatrix A(4, 1, 1, 0.0);
    A(0, 0) = 2.0;
    real* orig_data = A.data();

    BandedMatrix B(std::move(A));

    EXPECT_EQ(B.size(), 4);
    EXPECT_EQ(B(0, 0), 2.0);
    EXPECT_EQ(B.data(), orig_data);  // Same memory
}

// Tridiagonal System Tests (special case of banded)

TEST(BandedSolver, Tridiagonal4x4) {
    // Same system as Thomas algorithm test:
    // | 2 -1  0  0 | |x0|   | 1 |
    // |-1  2 -1  0 | |x1| = | 0 |
    // | 0 -1  2 -1 | |x2|   | 0 |
    // | 0  0 -1  2 | |x3|   | 1 |
    // Solution: x = [1, 1, 1, 1]

    BandedMatrix A(4, 1, 1, 0.0);

    // Set up tridiagonal system
    for (idx i = 0; i < 4; ++i) {
        A(i, i) = 2.0;
        if (i > 0) A(i, i - 1) = -1.0;
        if (i < 3) A(i, i + 1) = -1.0;
    }

    Vector b{1.0, 0.0, 0.0, 1.0};
    Vector x(4, 0.0);

    BandedSolverResult result = banded_solve(A, b, x);

    EXPECT_TRUE(result.success);
    EXPECT_NEAR(x[0], 1.0, 1e-10);
    EXPECT_NEAR(x[1], 1.0, 1e-10);
    EXPECT_NEAR(x[2], 1.0, 1e-10);
    EXPECT_NEAR(x[3], 1.0, 1e-10);
}

TEST(BandedSolver, Tridiagonal1DLaplacian) {
    // 1D Laplacian: -u'' = f with Dirichlet BC
    // Pattern: -1, 2, -1
    idx n = 20;
    BandedMatrix A(n, 1, 1, 0.0);

    for (idx i = 0; i < n; ++i) {
        A(i, i) = 2.0;
        if (i > 0) A(i, i - 1) = -1.0;
        if (i < n - 1) A(i, i + 1) = -1.0;
    }

    Vector b(n, 1.0);  // Constant RHS
    Vector x(n, 0.0);

    BandedSolverResult result = banded_solve(A, b, x);

    EXPECT_TRUE(result.success);

    // Verify solution by computing residual
    Vector r(n);
    banded_matvec(A, x, r);

    real max_err = 0.0;
    for (idx i = 0; i < n; ++i) {
        max_err = std::max(max_err, std::abs(r[i] - b[i]));
    }
    EXPECT_LT(max_err, 1e-10);
}

// Pentadiagonal System Tests

TEST(BandedSolver, Pentadiagonal) {
    // 2nd order compact finite difference stencil (pentadiagonal)
    idx n = 10;
    BandedMatrix A(n, 2, 2, 0.0);

    // Pattern: 1, -4, 6, -4, 1 (biharmonic operator)
    for (idx i = 0; i < n; ++i) {
        A(i, i) = 6.0;
        if (i > 0) A(i, i - 1) = -4.0;
        if (i > 1) A(i, i - 2) = 1.0;
        if (i < n - 1) A(i, i + 1) = -4.0;
        if (i < n - 2) A(i, i + 2) = 1.0;
    }

    // Make diagonally dominant by scaling
    for (idx i = 0; i < n; ++i) {
        A(i, i) = 10.0;  // Override for numerical stability
    }

    Vector b(n, 1.0);
    Vector x(n, 0.0);

    BandedSolverResult result = banded_solve(A, b, x);

    EXPECT_TRUE(result.success);

    // Verify residual
    Vector r(n);
    banded_matvec(A, x, r);

    real norm_r = 0.0;
    for (idx i = 0; i < n; ++i) {
        norm_r += (r[i] - b[i]) * (r[i] - b[i]);
    }
    EXPECT_LT(std::sqrt(norm_r), 1e-10);
}

// General Banded System Tests

TEST(BandedSolver, GeneralBanded) {
    // General banded system with kl=3, ku=2
    idx n = 15;
    BandedMatrix A(n, 3, 2, 0.0);

    // Create a diagonally dominant system
    for (idx j = 0; j < n; ++j) {
        real diag_sum = 0.0;
        for (idx i = (j > 2 ? j - 2 : 0); i < j; ++i) {
            A(i, j) = -0.1;
            diag_sum += 0.1;
        }
        for (idx i = j + 1; i <= std::min(j + 3, n - 1); ++i) {
            A(i, j) = -0.1;
            diag_sum += 0.1;
        }
        A(j, j) = diag_sum + 1.0;  // Diagonally dominant
    }

    Vector b(n);
    for (idx i = 0; i < n; ++i) {
        b[i] = static_cast<real>(i + 1);
    }

    Vector x(n, 0.0);

    BandedSolverResult result = banded_solve(A, b, x);

    EXPECT_TRUE(result.success);

    // Verify residual
    Vector r(n);
    banded_matvec(A, x, r);

    real norm_r = 0.0;
    real norm_b = 0.0;
    for (idx i = 0; i < n; ++i) {
        norm_r += (r[i] - b[i]) * (r[i] - b[i]);
        norm_b += b[i] * b[i];
    }
    EXPECT_LT(std::sqrt(norm_r) / std::sqrt(norm_b), 1e-10);
}

// LU Factorization and Reuse Tests

TEST(BandedSolver, LUFactorizationReuse) {
    // Test that we can factor once and solve multiple times
    idx n = 10;
    BandedMatrix A(n, 1, 1, 0.0);

    for (idx i = 0; i < n; ++i) {
        A(i, i) = 4.0;
        if (i > 0) A(i, i - 1) = -1.0;
        if (i < n - 1) A(i, i + 1) = -1.0;
    }

    // Keep original for verification
    BandedMatrix A_orig = A;

    // Factor
    std::unique_ptr<idx[]> ipiv = std::make_unique<idx[]>(n);
    BandedSolverResult result = banded_lu(A, ipiv.get());
    EXPECT_TRUE(result.success);

    // Solve with different RHS vectors
    for (int trial = 0; trial < 5; ++trial) {
        Vector b(n);
        for (idx i = 0; i < n; ++i) {
            b[i] = static_cast<real>((trial + 1) * (i + 1));
        }

        Vector x = b;  // Copy RHS
        banded_lu_solve(A, ipiv.get(), x);

        // Verify with original matrix
        Vector r(n);
        banded_matvec(A_orig, x, r);

        real max_err = 0.0;
        for (idx i = 0; i < n; ++i) {
            max_err = std::max(max_err, std::abs(r[i] - b[i]));
        }
        EXPECT_LT(max_err, 1e-10);
    }
}

TEST(BandedSolver, MultipleRHS) {
    // Test solving with multiple right-hand sides at once
    idx n = 8;
    idx nrhs = 4;

    BandedMatrix A(n, 1, 1, 0.0);
    for (idx i = 0; i < n; ++i) {
        A(i, i) = 3.0;
        if (i > 0) A(i, i - 1) = -1.0;
        if (i < n - 1) A(i, i + 1) = -1.0;
    }

    BandedMatrix A_orig = A;

    // Factor
    std::unique_ptr<idx[]> ipiv = std::make_unique<idx[]>(n);
    BandedSolverResult result = banded_lu(A, ipiv.get());
    EXPECT_TRUE(result.success);

    // Multiple RHS (column-major)
    std::unique_ptr<real[]> B = std::make_unique<real[]>(n * nrhs);
    for (idx rhs = 0; rhs < nrhs; ++rhs) {
        for (idx i = 0; i < n; ++i) {
            B[i + rhs * n] = static_cast<real>((rhs + 1) * (i + 1));
        }
    }

    // Keep copy for verification
    std::unique_ptr<real[]> B_orig = std::make_unique<real[]>(n * nrhs);
    std::memcpy(B_orig.get(), B.get(), n * nrhs * sizeof(real));

    // Solve all at once
    banded_lu_solve_multi(A, ipiv.get(), B.get(), nrhs);

    // Verify each solution
    for (idx rhs = 0; rhs < nrhs; ++rhs) {
        Vector x(n);
        Vector b(n);
        for (idx i = 0; i < n; ++i) {
            x[i] = B[i + rhs * n];
            b[i] = B_orig[i + rhs * n];
        }

        Vector r(n);
        banded_matvec(A_orig, x, r);

        real max_err = 0.0;
        for (idx i = 0; i < n; ++i) {
            max_err = std::max(max_err, std::abs(r[i] - b[i]));
        }
        EXPECT_LT(max_err, 1e-10);
    }
}

// Matrix-Vector Product Tests

TEST(BandedMatvec, Basic) {
    BandedMatrix A(4, 1, 1, 0.0);
    A(0, 0) = 2.0; A(0, 1) = -1.0;
    A(1, 0) = -1.0; A(1, 1) = 2.0; A(1, 2) = -1.0;
    A(2, 1) = -1.0; A(2, 2) = 2.0; A(2, 3) = -1.0;
    A(3, 2) = -1.0; A(3, 3) = 2.0;

    Vector x{1.0, 2.0, 3.0, 4.0};
    Vector y(4);

    banded_matvec(A, x, y);

    // Manual calculation:
    // y[0] = 2*1 - 1*2 = 0
    // y[1] = -1*1 + 2*2 - 1*3 = 0
    // y[2] = -1*2 + 2*3 - 1*4 = 0
    // y[3] = -1*3 + 2*4 = 5
    EXPECT_NEAR(y[0], 0.0, 1e-10);
    EXPECT_NEAR(y[1], 0.0, 1e-10);
    EXPECT_NEAR(y[2], 0.0, 1e-10);
    EXPECT_NEAR(y[3], 5.0, 1e-10);
}

TEST(BandedMatvec, GEMV) {
    BandedMatrix A(3, 1, 1, 0.0);
    A(0, 0) = 1.0; A(0, 1) = 2.0;
    A(1, 0) = 3.0; A(1, 1) = 4.0; A(1, 2) = 5.0;
    A(2, 1) = 6.0; A(2, 2) = 7.0;

    Vector x{1.0, 1.0, 1.0};
    Vector y{10.0, 20.0, 30.0};

    // y = 2*A*x + 3*y
    banded_gemv(2.0, A, x, 3.0, y);

    // A*x = [3, 12, 13]
    // y = 2*[3,12,13] + 3*[10,20,30] = [6,24,26] + [30,60,90] = [36,84,116]
    EXPECT_NEAR(y[0], 36.0, 1e-10);
    EXPECT_NEAR(y[1], 84.0, 1e-10);
    EXPECT_NEAR(y[2], 116.0, 1e-10);
}

// Large System Tests (for HPC validation)

TEST(BandedSolver, LargeTridiagonal) {
    // Large system to verify correctness at scale
    idx n = 10000;
    BandedMatrix A(n, 1, 1, 0.0);

    // 1D Laplacian
    for (idx i = 0; i < n; ++i) {
        A(i, i) = 2.0;
        if (i > 0) A(i, i - 1) = -1.0;
        if (i < n - 1) A(i, i + 1) = -1.0;
    }

    Vector b(n, 1.0);
    Vector x(n, 0.0);

    BandedSolverResult result = banded_solve(A, b, x);

    EXPECT_TRUE(result.success);

    // Spot check residual at several points
    // Tolerance relaxed for large systems due to floating-point accumulation
    Vector r(n);
    banded_matvec(A, x, r);

    real max_err = 0.0;
    for (idx i = 0; i < n; i += 100) {
        max_err = std::max(max_err, std::abs(r[i] - b[i]));
    }
    EXPECT_LT(max_err, 1e-8);  // 8 digits of accuracy for n=10000
}

TEST(BandedSolver, LargePentadiagonal) {
    // Large pentadiagonal system
    idx n = 5000;
    BandedMatrix A(n, 2, 2, 0.0);

    // Diagonally dominant pentadiagonal
    for (idx i = 0; i < n; ++i) {
        A(i, i) = 10.0;
        if (i > 0) A(i, i - 1) = -2.0;
        if (i > 1) A(i, i - 2) = -0.5;
        if (i < n - 1) A(i, i + 1) = -2.0;
        if (i < n - 2) A(i, i + 2) = -0.5;
    }

    Vector b(n, 1.0);
    Vector x(n, 0.0);

    BandedSolverResult result = banded_solve(A, b, x);

    EXPECT_TRUE(result.success);

    // Verify residual
    Vector r(n);
    banded_matvec(A, x, r);

    real norm_r = 0.0;
    real norm_b = 0.0;
    for (idx i = 0; i < n; ++i) {
        norm_r += (r[i] - b[i]) * (r[i] - b[i]);
        norm_b += b[i] * b[i];
    }
    EXPECT_LT(std::sqrt(norm_r) / std::sqrt(norm_b), 1e-10);
}

// Condition Number and Norm Tests

TEST(BandedNorm, Norm1) {
    BandedMatrix A(3, 1, 1, 0.0);
    A(0, 0) = 1.0; A(0, 1) = 2.0;
    A(1, 0) = 3.0; A(1, 1) = 4.0; A(1, 2) = 5.0;
    A(2, 1) = 6.0; A(2, 2) = 7.0;

    // Column sums: [1+3, 2+4+6, 5+7] = [4, 12, 12]
    // Max = 12
    real norm = banded_norm1(A);
    EXPECT_NEAR(norm, 12.0, 1e-10);
}

// Edge Cases

TEST(BandedSolver, Size1) {
    BandedMatrix A(1, 0, 0, 0.0);
    A(0, 0) = 5.0;

    Vector b{10.0};
    Vector x(1, 0.0);

    BandedSolverResult result = banded_solve(A, b, x);

    EXPECT_TRUE(result.success);
    EXPECT_NEAR(x[0], 2.0, 1e-10);
}

TEST(BandedSolver, Size2) {
    BandedMatrix A(2, 1, 1, 0.0);
    A(0, 0) = 3.0; A(0, 1) = 1.0;
    A(1, 0) = 2.0; A(1, 1) = 4.0;

    // System: 3x + y = 5, 2x + 4y = 6
    // Solution: x = 1.4, y = 0.8
    Vector b{5.0, 6.0};
    Vector x(2, 0.0);

    BandedSolverResult result = banded_solve(A, b, x);

    EXPECT_TRUE(result.success);
    EXPECT_NEAR(x[0], 1.4, 1e-10);
    EXPECT_NEAR(x[1], 0.8, 1e-10);
}

TEST(BandedSolver, DiagonalMatrix) {
    // Pure diagonal (kl=ku=0)
    idx n = 5;
    BandedMatrix A(n, 0, 0, 0.0);

    for (idx i = 0; i < n; ++i) {
        A(i, i) = static_cast<real>(i + 1);
    }

    Vector b{1.0, 2.0, 3.0, 4.0, 5.0};
    Vector x(n, 0.0);

    BandedSolverResult result = banded_solve(A, b, x);

    EXPECT_TRUE(result.success);
    for (idx i = 0; i < n; ++i) {
        EXPECT_NEAR(x[i], 1.0, 1e-10);  // Solution is all ones
    }
}
