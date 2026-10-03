/// @file test_banded.cpp
/// @brief Tests for banded matrix solver

#include "linear/banded/banded.hpp"
#include "linear/factorization/lu.hpp"
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <random>

using namespace num;

// band_mat Construction and Access Tests

TEST(band_mat, ConstructBasic) {
    band_mat A(10, 2, 3); // 10x10 with 2 lower, 3 upper diagonals

    EXPECT_EQ(A.size(), 10);
    EXPECT_EQ(A.rows(), 10);
    EXPECT_EQ(A.cols(), 10);
    EXPECT_EQ(A.kl(), 2);
    EXPECT_EQ(A.ku(), 3);
    EXPECT_EQ(A.bandwidth(), 6); // kl + ku + 1
    EXPECT_EQ(A.ldab(), 8);      // 2*kl + ku + 1
}

TEST(band_mat, ConstructWithValue) {
    band_mat A(5, 1, 1, 2.0);

    // Check that band elements are initialized
    for (idx j = 0; j < 5; ++j) {
        for (idx i = (j > 0 ? j - 1 : 0); i <= std::min(j + 1, idx(4)); ++i) {
            EXPECT_EQ(A(i, j), 2.0);
        }
    }
}

TEST(band_mat, ElementAccess) {
    band_mat A(5, 1, 2, 0.0); // tridiagonal plus one extra upper

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

TEST(band_mat, InBandCheck) {
    band_mat A(5, 1, 2, 0.0);

    // Diagonal is in band
    EXPECT_TRUE(A.in_band(2, 2));

    // Lower diagonal is in band
    EXPECT_TRUE(A.in_band(3, 2));

    // Upper diagonals in band
    EXPECT_TRUE(A.in_band(2, 3));
    EXPECT_TRUE(A.in_band(2, 4));

    // Outside band
    EXPECT_FALSE(A.in_band(0, 3)); // Too far above
    EXPECT_FALSE(A.in_band(4, 0)); // Too far below
}

TEST(band_mat, CopyConstruct) {
    band_mat A(4, 1, 1, 0.0);
    A(0, 0) = 2.0;
    A(0, 1) = -1.0;
    A(1, 0) = -1.0;
    A(1, 1) = 2.0;

    band_mat B(A);

    EXPECT_EQ(B.size(), 4);
    EXPECT_EQ(B.kl(), 1);
    EXPECT_EQ(B.ku(), 1);
    EXPECT_EQ(B(0, 0), 2.0);
    EXPECT_EQ(B(0, 1), -1.0);

    // Ensure deep copy
    A(0, 0) = 999.0;
    EXPECT_EQ(B(0, 0), 2.0);
}

TEST(band_mat, MoveConstruct) {
    band_mat A(4, 1, 1, 0.0);
    A(0, 0) = 2.0;
    real *orig_data = A.data();

    band_mat B(std::move(A));

    EXPECT_EQ(B.size(), 4);
    EXPECT_EQ(B(0, 0), 2.0);
    EXPECT_EQ(B.data(), orig_data); // Same memory
}

// tridiagonal System Tests (special case of banded)

TEST(BandedSolver, Tridiagonal4x4) {
    // Same system as Thomas algorithm test:
    // | 2 -1  0  0 | |x0|   | 1 |
    // |-1  2 -1  0 | |x1| = | 0 |
    // | 0 -1  2 -1 | |x2|   | 0 |
    // | 0  0 -1  2 | |x3|   | 1 |
    // Solution: x = [1, 1, 1, 1]

    band_mat A(4, 1, 1, 0.0);

    // Set up tridiagonal system
    for (idx i = 0; i < 4; ++i) {
        A(i, i) = 2.0;
        if (i > 0) {
            A(i, i - 1) = -1.0;
        }
        if (i < 3) {
            A(i, i + 1) = -1.0;
        }
    }

    vec<real> b{1.0, 0.0, 0.0, 1.0};
    vec<real> x(4, 0.0);

    const banded_lu_result factor = lu(A);

    EXPECT_FALSE(factor.singular);
    solve(factor, b, x);
    EXPECT_NEAR(x[0], 1.0, 1e-10);
    EXPECT_NEAR(x[1], 1.0, 1e-10);
    EXPECT_NEAR(x[2], 1.0, 1e-10);
    EXPECT_NEAR(x[3], 1.0, 1e-10);
}

TEST(BandedSolver, Tridiagonal1DLaplacian) {
    // 1D Laplacian: -u'' = f with Dirichlet BC
    // Pattern: -1, 2, -1
    idx n = 20;
    band_mat A(n, 1, 1, 0.0);

    for (idx i = 0; i < n; ++i) {
        A(i, i) = 2.0;
        if (i > 0) {
            A(i, i - 1) = -1.0;
        }
        if (i < n - 1) {
            A(i, i + 1) = -1.0;
        }
    }

    vec<real> b(n, 1.0); // Constant RHS
    vec<real> x(n, 0.0);

    const banded_lu_result factor = lu(A);

    EXPECT_FALSE(factor.singular);
    solve(factor, b, x);

    // Verify solution by computing residual
    vec<real> r(n);
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
    band_mat A(n, 2, 2, 0.0);

    // Pattern: 1, -4, 6, -4, 1 (biharmonic operator)
    for (idx i = 0; i < n; ++i) {
        A(i, i) = 6.0;
        if (i > 0) {
            A(i, i - 1) = -4.0;
        }
        if (i > 1) {
            A(i, i - 2) = 1.0;
        }
        if (i < n - 1) {
            A(i, i + 1) = -4.0;
        }
        if (i < n - 2) {
            A(i, i + 2) = 1.0;
        }
    }

    // Make diagonally dominant by scaling
    for (idx i = 0; i < n; ++i) {
        A(i, i) = 10.0; // Override for numerical stability
    }

    vec<real> b(n, 1.0);
    vec<real> x(n, 0.0);

    const banded_lu_result factor = lu(A);

    EXPECT_FALSE(factor.singular);
    solve(factor, b, x);

    // Verify residual
    vec<real> r(n);
    banded_matvec(A, x, r);

    real norm_r = 0.0;
    for (idx i = 0; i < n; ++i) {
        norm_r += (r[i] - b[i]) * (r[i] - b[i]);
    }
    EXPECT_LT(std::sqrt(norm_r), 1e-10);
}

// General banded System Tests

TEST(BandedSolver, GeneralBanded) {
    // General banded system with kl=3, ku=2
    idx n = 15;
    band_mat A(n, 3, 2, 0.0);

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
        A(j, j) = diag_sum + 1.0; // Diagonally dominant
    }

    vec<real> b(n);
    for (idx i = 0; i < n; ++i) {
        b[i] = static_cast<real>(i + 1);
    }

    vec<real> x(n, 0.0);

    const banded_lu_result factor = lu(A);

    EXPECT_FALSE(factor.singular);
    solve(factor, b, x);

    // Verify residual
    vec<real> r(n);
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
    band_mat A(n, 1, 1, 0.0);

    for (idx i = 0; i < n; ++i) {
        A(i, i) = 4.0;
        if (i > 0) {
            A(i, i - 1) = -1.0;
        }
        if (i < n - 1) {
            A(i, i + 1) = -1.0;
        }
    }

    // Keep original for verification
    band_mat A_orig = A;

    // Factor
    const banded_lu_result factor = lu(A);
    EXPECT_FALSE(factor.singular);

    // Solve with different RHS vectors
    for (int trial = 0; trial < 5; ++trial) {
        vec<real> b(n);
        for (idx i = 0; i < n; ++i) {
            b[i] = static_cast<real>((trial + 1) * (i + 1));
        }

        vec<real> x;
        solve(factor, b, x);

        // Verify with original matrix
        vec<real> r(n);
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

    band_mat A(n, 1, 1, 0.0);
    for (idx i = 0; i < n; ++i) {
        A(i, i) = 3.0;
        if (i > 0) {
            A(i, i - 1) = -1.0;
        }
        if (i < n - 1) {
            A(i, i + 1) = -1.0;
        }
    }

    band_mat A_orig = A;

    // Factor
    const banded_lu_result factor = lu(A);
    EXPECT_FALSE(factor.singular);

    // One right-hand side per column.
    mat<real> B(n, nrhs, 0.0);
    for (idx rhs = 0; rhs < nrhs; ++rhs) {
        for (idx i = 0; i < n; ++i) {
            B(i, rhs) = static_cast<real>((rhs + 1) * (i + 1));
        }
    }

    mat<real> X;
    solve(factor, B, X);

    // Verify each solution
    for (idx rhs = 0; rhs < nrhs; ++rhs) {
        vec<real> x(n);
        vec<real> b(n);
        for (idx i = 0; i < n; ++i) {
            x[i] = X(i, rhs);
            b[i] = B(i, rhs);
        }

        vec<real> r(n);
        banded_matvec(A_orig, x, r);

        real max_err = 0.0;
        for (idx i = 0; i < n; ++i) {
            max_err = std::max(max_err, std::abs(r[i] - b[i]));
        }
        EXPECT_LT(max_err, 1e-10);
    }
}

// mat-vec Product Tests

TEST(BandedMatvec, Basic) {
    band_mat A(4, 1, 1, 0.0);
    A(0, 0) = 2.0;
    A(0, 1) = -1.0;
    A(1, 0) = -1.0;
    A(1, 1) = 2.0;
    A(1, 2) = -1.0;
    A(2, 1) = -1.0;
    A(2, 2) = 2.0;
    A(2, 3) = -1.0;
    A(3, 2) = -1.0;
    A(3, 3) = 2.0;

    vec<real> x{1.0, 2.0, 3.0, 4.0};
    vec<real> y(4);

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
    band_mat A(3, 1, 1, 0.0);
    A(0, 0) = 1.0;
    A(0, 1) = 2.0;
    A(1, 0) = 3.0;
    A(1, 1) = 4.0;
    A(1, 2) = 5.0;
    A(2, 1) = 6.0;
    A(2, 2) = 7.0;

    vec<real> x{1.0, 1.0, 1.0};
    vec<real> y{10.0, 20.0, 30.0};

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
    band_mat A(n, 1, 1, 0.0);

    // 1D Laplacian
    for (idx i = 0; i < n; ++i) {
        A(i, i) = 2.0;
        if (i > 0) {
            A(i, i - 1) = -1.0;
        }
        if (i < n - 1) {
            A(i, i + 1) = -1.0;
        }
    }

    vec<real> b(n, 1.0);
    vec<real> x(n, 0.0);

    const banded_lu_result factor = lu(A);

    EXPECT_FALSE(factor.singular);
    solve(factor, b, x);

    // Spot check residual at several points
    // Tolerance relaxed for large systems due to floating-point accumulation
    vec<real> r(n);
    banded_matvec(A, x, r);

    real max_err = 0.0;
    for (idx i = 0; i < n; i += 100) {
        max_err = std::max(max_err, std::abs(r[i] - b[i]));
    }
    EXPECT_LT(max_err, 1e-8); // 8 digits of accuracy for n=10000
}

TEST(BandedSolver, LargePentadiagonal) {
    // Large pentadiagonal system
    idx n = 5000;
    band_mat A(n, 2, 2, 0.0);

    // Diagonally dominant pentadiagonal
    for (idx i = 0; i < n; ++i) {
        A(i, i) = 10.0;
        if (i > 0) {
            A(i, i - 1) = -2.0;
        }
        if (i > 1) {
            A(i, i - 2) = -0.5;
        }
        if (i < n - 1) {
            A(i, i + 1) = -2.0;
        }
        if (i < n - 2) {
            A(i, i + 2) = -0.5;
        }
    }

    vec<real> b(n, 1.0);
    vec<real> x(n, 0.0);

    const banded_lu_result factor = lu(A);

    EXPECT_FALSE(factor.singular);
    solve(factor, b, x);

    // Verify residual
    vec<real> r(n);
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
    band_mat A(3, 1, 1, 0.0);
    A(0, 0) = 1.0;
    A(0, 1) = 2.0;
    A(1, 0) = 3.0;
    A(1, 1) = 4.0;
    A(1, 2) = 5.0;
    A(2, 1) = 6.0;
    A(2, 2) = 7.0;

    // Column sums: [1+3, 2+4+6, 5+7] = [4, 12, 12]
    // Max = 12
    real norm = banded_norm1(A);
    EXPECT_NEAR(norm, 12.0, 1e-10);
}

// graph_edge Cases

TEST(BandedSolver, Size1) {
    band_mat A(1, 0, 0, 0.0);
    A(0, 0) = 5.0;

    vec<real> b{10.0};
    vec<real> x(1, 0.0);

    const banded_lu_result factor = lu(A);

    EXPECT_FALSE(factor.singular);
    solve(factor, b, x);
    EXPECT_NEAR(x[0], 2.0, 1e-10);
}

TEST(BandedSolver, Size2) {
    band_mat A(2, 1, 1, 0.0);
    A(0, 0) = 3.0;
    A(0, 1) = 1.0;
    A(1, 0) = 2.0;
    A(1, 1) = 4.0;

    // System: 3x + y = 5, 2x + 4y = 6
    // Solution: x = 1.4, y = 0.8
    vec<real> b{5.0, 6.0};
    vec<real> x(2, 0.0);

    const banded_lu_result factor = lu(A);

    EXPECT_FALSE(factor.singular);
    solve(factor, b, x);
    EXPECT_NEAR(x[0], 1.4, 1e-10);
    EXPECT_NEAR(x[1], 0.8, 1e-10);
}

TEST(BandedSolver, DiagonalMatrix) {
    // Pure diagonal (kl=ku=0)
    idx n = 5;
    band_mat A(n, 0, 0, 0.0);

    for (idx i = 0; i < n; ++i) {
        A(i, i) = static_cast<real>(i + 1);
    }

    vec<real> b{1.0, 2.0, 3.0, 4.0, 5.0};
    vec<real> x(n, 0.0);

    const banded_lu_result factor = lu(A);

    EXPECT_FALSE(factor.singular);
    solve(factor, b, x);
    for (idx i = 0; i < n; ++i) {
        EXPECT_NEAR(x[i], 1.0, 1e-10); // Solution is all ones
    }
}

// A band matrix with small diagonal entries, so partial pivoting swaps rows and the swapped
// rows carry fill beyond the upper bandwidth. Partial pivoting is backward stable, so both
// solves must leave a residual at rounding level relative to |A| |x|. Some of these matrices
// are too ill-conditioned for a forward-error comparison against dense LU to be meaningful.
TEST(BandedSolver, PivotingIsBackwardStable) {
    for (idx kl : {1, 2, 3}) {
        for (idx ku : {0, 1, 2}) {
            const idx n = 12;
            band_mat A(n, kl, ku, 0.0);
            std::mt19937 rng(static_cast<unsigned>(17 + 5 * kl + ku));
            std::uniform_real_distribution<real> entry(0.5, 2.0);
            real norm_A = 0.0;
            for (idx i = 0; i < n; ++i) {
                real row = 0.0;
                for (idx j = (i > kl ? i - kl : 0); j <= std::min(i + ku, n - 1); ++j) {
                    A(i, j) = (i == j) ? 1e-3 * entry(rng) : entry(rng);
                    row += std::abs(A(i, j));
                }
                norm_A = std::max(norm_A, row);
            }
            vec<real> b(n);
            mat<real> B(n, 1, 0.0);
            for (idx i = 0; i < n; ++i) {
                b[i] = B(i, 0) = static_cast<real>(i) - 4.0;
            }

            const banded_lu_result factor = lu(A);
            ASSERT_FALSE(factor.singular) << "kl=" << kl << " ku=" << ku;
            vec<real> x;
            mat<real> X;
            solve(factor, b, x);
            solve(factor, B, X);

            vec<real> X_column(n);
            for (idx i = 0; i < n; ++i) {
                X_column[i] = X(i, 0);
            }
            for (const vec<real> *solution : {&x, &X_column}) {
                vec<real> Ax(n);
                banded_matvec(A, *solution, Ax);
                real residual = 0.0, norm_x = 0.0, norm_b = 0.0;
                for (idx i = 0; i < n; ++i) {
                    residual = std::max(residual, std::abs(b[i] - Ax[i]));
                    norm_x = std::max(norm_x, std::abs((*solution)[i]));
                    norm_b = std::max(norm_b, std::abs(b[i]));
                }
                EXPECT_LT(residual / (norm_A * norm_x + norm_b), 1e-13)
                    << "kl=" << kl << " ku=" << ku;
            }
        }
    }
}

// Well-conditioned but still pivoting: the band and dense solutions agree.
TEST(BandedSolver, PivotingMatchesDenseLU) {
    const idx n = 10, kl = 2, ku = 1;
    band_mat A(n, kl, ku, 0.0);
    mat<real> dense(n, n, 0.0);
    for (idx i = 0; i < n; ++i) {
        for (idx j = (i > kl ? i - kl : 0); j <= std::min(i + ku, n - 1); ++j) {
            // The largest entry of each column sits below the diagonal, forcing a swap.
            const real value = (i == j + 1) ? 4.0 : (i == j ? 1.0 : 0.5);
            A(i, j) = dense(i, j) = value;
        }
    }
    vec<real> b(n), x, expected;
    for (idx i = 0; i < n; ++i) {
        b[i] = static_cast<real>(i + 1);
    }
    const banded_lu_result factor = lu(A);
    ASSERT_FALSE(factor.singular);
    idx swaps = 0;
    for (idx k = 0; k < n; ++k) {
        swaps += factor.swaps[k] != k ? 1 : 0;
    }
    EXPECT_GT(swaps, 0);
    solve(factor, b, x);
    solve(lu(dense), b, expected);
    for (idx i = 0; i < n; ++i) {
        EXPECT_NEAR(x[i], expected[i], 1e-10 * (1.0 + std::abs(expected[i])));
    }
}
