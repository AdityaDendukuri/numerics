#include "linear/factorization/factorization.hpp"
#include "container/matrix_ops.hpp"
#include "linear/matrix_utils.hpp"
#include "linear/sparse/klu.hpp"
#include "linear/sparse/sparse.hpp"
#include "linear/sparse/umfpack.hpp"
#include <cmath>
#include <gtest/gtest.h>

using namespace num;

// Helpers
static real mat_norm_inf(const mat &A) {
    real m = 0;
    for (idx i = 0; i < A.rows(); ++i) {
        for (idx j = 0; j < A.cols(); ++j) {
            m = std::max(m, std::abs(A(i, j)));
        }
    }
    return m;
}

static real vec_norm_inf(const vec &v) {
    real m = 0;
    for (idx i = 0; i < v.size(); ++i) {
        m = std::max(m, std::abs(v[i]));
    }
    return m;
}

static mat low_rank_update(const mat &matrix, const mat &vectors, const mat &weights) {
    mat updated = matrix;
    for (idx row = 0; row < matrix.rows(); ++row) {
        for (idx column = 0; column < matrix.cols(); ++column) {
            for (idx left = 0; left < vectors.cols(); ++left) {
                for (idx right = 0; right < vectors.cols(); ++right) {
                    updated(row, column) +=
                        vectors(row, left) * weights(left, right) * vectors(column, right);
                }
            }
        }
    }
    return updated;
}

// LU factorization

TEST(KLU, SparseFactorAndBlockSolve) {
    if (!klu_available()) {
        GTEST_SKIP() << "SuiteSparse KLU is not available";
    }

    const auto A = spmat::from_triplets(3, 3, {0, 0, 1, 1, 1, 2, 2}, {0, 1, 0, 1, 2, 1, 2},
                                               {4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 3.0});
    klu_factorization factor(A);
    vec b{15.0, 10.0, 10.0};
    vec x;
    factor.solve(b, x);
    EXPECT_NEAR(x[0], 5.0, 1e-12);
    EXPECT_NEAR(x[1], 5.0, 1e-12);
    EXPECT_NEAR(x[2], 5.0, 1e-12);

    mat B(3, 2, 0.0);
    for (idx row = 0; row < 3; ++row) {
        B(row, 0) = b[row];
        B(row, 1) = 2.0 * b[row];
    }
    mat X;
    factor.solve(B, X);
    for (idx row = 0; row < 3; ++row) {
        EXPECT_NEAR(X(row, 0), 5.0, 1e-12);
        EXPECT_NEAR(X(row, 1), 10.0, 1e-12);
    }
}

TEST(Cholesky, RankOneUpdateAndDowndate) {
    mat identity_matrix = identity(3);
    auto factor = cholesky(assume_spd(identity_matrix));
    vec update{0.5, -0.25, 0.75};
    cholesky_update(factor, update);

    mat reconstructed(3, 3, 0.0);
    for (idx row = 0; row < 3; ++row) {
        for (idx column = 0; column < 3; ++column) {
            for (idx k = 0; k < 3; ++k) {
                reconstructed(row, column) += factor.L(row, k) * factor.L(column, k);
            }
            EXPECT_NEAR(reconstructed(row, column),
                        (row == column ? 1.0 : 0.0) + update[row] * update[column], 1e-12);
        }
    }

    cholesky_downdate(factor, update);
    for (idx row = 0; row < 3; ++row) {
        for (idx column = 0; column <= row; ++column) {
            EXPECT_NEAR(factor.L(row, column), row == column ? 1.0 : 0.0, 1e-12);
        }
    }
}

TEST(AutoLinear, SolveTransposeAndInverseDiagonal) {
    const auto matrix =
        spmat::from_triplets(2, 2, {0, 0, 1, 1}, {0, 1, 0, 1}, {4.0, 1.0, 2.0, 3.0});
    auto_linear_solver factor(matrix, {.dense_limit = 0});

    vec solution(2, 0.0);
    factor.solve(vec{6.0, 8.0}, solution);
    EXPECT_NEAR(solution[0], 1.0, 1e-12);
    EXPECT_NEAR(solution[1], 2.0, 1e-12);

    factor.solve_transpose(vec{8.0, 7.0}, solution);
    EXPECT_NEAR(solution[0], 1.0, 1e-12);
    EXPECT_NEAR(solution[1], 2.0, 1e-12);

    vec inverse_diag(2, 0.0);
    inverse_diagonal_workspace workspace;
    inverse_diagonal(factor, inverse_diag, workspace, 1);
    EXPECT_NEAR(inverse_diag[0], 0.3, 1e-12);
    EXPECT_NEAR(inverse_diag[1], 0.4, 1e-12);

    const std::vector<idx> rows{0, 1};
    const std::vector<idx> columns{1, 0};
    vec selected(2, 0.0);
    selected_inverse(factor, rows, columns, selected, workspace);
    EXPECT_NEAR(selected[0], -0.1, 1e-12);
    EXPECT_NEAR(selected[1], -0.2, 1e-12);

    const std::vector<idx> indices{1, 0};
    mat principal;
    inverse_principal_block(factor, indices, principal, workspace);
    EXPECT_NEAR(principal(0, 0), 0.4, 1e-12);
    EXPECT_NEAR(principal(0, 1), -0.2, 1e-12);
    EXPECT_NEAR(principal(1, 0), -0.1, 1e-12);
    EXPECT_NEAR(principal(1, 1), 0.3, 1e-12);
}

TEST(AutoLinear, ConvenienceOverloadsMatchOutParameterForm) {
    const auto matrix =
        spmat::from_triplets(2, 2, {0, 0, 1, 1}, {0, 1, 0, 1}, {4.0, 1.0, 2.0, 3.0});
    auto_linear_solver factor(matrix, {.dense_limit = 0});

    const vec b{6.0, 8.0};
    mat B(2, 2, 0.0);
    B(0, 0) = 6.0;
    B(0, 1) = 1.0;
    B(1, 0) = 8.0;
    B(1, 1) = 5.0;

    vec expected_vector(2, 0.0);
    factor.solve(b, expected_vector);
    const vec actual_vector = solve(factor, b);
    EXPECT_DOUBLE_EQ(actual_vector[0], expected_vector[0]);
    EXPECT_DOUBLE_EQ(actual_vector[1], expected_vector[1]);

    factor.solve_transpose(b, expected_vector);
    const vec actual_transpose = solve_transpose(factor, b);
    EXPECT_DOUBLE_EQ(actual_transpose[0], expected_vector[0]);
    EXPECT_DOUBLE_EQ(actual_transpose[1], expected_vector[1]);

    mat expected_matrix;
    factor.solve(B, expected_matrix);
    const mat actual_matrix = solve(factor, B);
    ASSERT_EQ(actual_matrix.rows(), expected_matrix.rows());
    ASSERT_EQ(actual_matrix.cols(), expected_matrix.cols());
    for (idx i = 0; i < expected_matrix.rows(); ++i) {
        for (idx j = 0; j < expected_matrix.cols(); ++j) {
            EXPECT_DOUBLE_EQ(actual_matrix(i, j), expected_matrix(i, j));
        }
    }

    factor.solve_transpose(B, expected_matrix);
    const mat actual_matrix_transpose = solve_transpose(factor, B);
    for (idx i = 0; i < expected_matrix.rows(); ++i) {
        for (idx j = 0; j < expected_matrix.cols(); ++j) {
            EXPECT_DOUBLE_EQ(actual_matrix_transpose(i, j), expected_matrix(i, j));
        }
    }

    vec expected_diagonal(2, 0.0);
    inverse_diagonal_workspace workspace;
    inverse_diagonal(factor, expected_diagonal, workspace);
    const vec actual_diagonal = inverse_diagonal(factor);
    EXPECT_DOUBLE_EQ(actual_diagonal[0], expected_diagonal[0]);
    EXPECT_DOUBLE_EQ(actual_diagonal[1], expected_diagonal[1]);

    const std::vector<idx> indices{1, 0};
    mat expected_block;
    inverse_principal_block(factor, indices, expected_block, workspace);
    const mat actual_block = inverse_principal_block(factor, indices);
    ASSERT_EQ(actual_block.rows(), expected_block.rows());
    ASSERT_EQ(actual_block.cols(), expected_block.cols());
    for (idx i = 0; i < expected_block.rows(); ++i) {
        for (idx j = 0; j < expected_block.cols(); ++j) {
            EXPECT_DOUBLE_EQ(actual_block(i, j), expected_block(i, j));
        }
    }
}

TEST(InversePrincipalBlock, DenseFactorizationsAndValidation) {
    mat matrix(3, 3, 0.0);
    matrix(0, 0) = 4.0;
    matrix(0, 1) = 1.0;
    matrix(1, 0) = 1.0;
    matrix(1, 1) = 3.0;
    matrix(1, 2) = 1.0;
    matrix(2, 1) = 1.0;
    matrix(2, 2) = 2.0;

    const auto lu_factor = lu(assume_square(matrix));
    const auto cholesky_factor = cholesky(assume_spd(matrix));
    const mat inverse = lu_inv(lu_factor);
    const std::vector<idx> indices{2, 0};
    inverse_diagonal_workspace workspace;

    for (const bool use_cholesky : {false, true}) {
        mat principal;
        if (use_cholesky) {
            inverse_principal_block(cholesky_factor, indices, principal, workspace);
        } else {
            inverse_principal_block(lu_factor, indices, principal, workspace);
        }
        for (idx row = 0; row < indices.size(); ++row) {
            for (idx column = 0; column < indices.size(); ++column) {
                EXPECT_NEAR(principal(row, column), inverse(indices[row], indices[column]), 1e-12);
            }
        }
    }

    mat principal;
    EXPECT_THROW(inverse_principal_block(lu_factor, std::vector<idx>{0, 0}, principal, workspace),
                 std::invalid_argument);
    EXPECT_THROW(inverse_principal_block(lu_factor, std::vector<idx>{3}, principal, workspace),
                 std::out_of_range);
}

TEST(UMFPACK, SparseFactorAndSolve) {
    if (!umfpack_available()) {
        GTEST_SKIP() << "SuiteSparse UMFPACK is not available";
    }
    const auto A = spmat::from_triplets(3, 3, {0, 0, 1, 1, 1, 2, 2}, {0, 1, 0, 1, 2, 1, 2},
                                               {4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 3.0});
    umfpack_factor factor(A);
    vec x;
    factor.solve(vec{15.0, 10.0, 10.0}, x);
    EXPECT_NEAR(x[0], 5.0, 1e-12);
    EXPECT_NEAR(x[1], 5.0, 1e-12);
    EXPECT_NEAR(x[2], 5.0, 1e-12);
}

TEST(LU, SolveSmall3x3) {
    // [2  1  0]       [1]
    // [4  3  2] x  =  [2]
    // [8  7  9]       [3]
    // solution: x = [-1/2, 2, -1/2]  -- computed by hand
    mat A(3, 3, 0.0);
    A(0, 0) = 2;
    A(0, 1) = 1;
    A(1, 0) = 4;
    A(1, 1) = 3;
    A(1, 2) = 2;
    A(2, 0) = 8;
    A(2, 1) = 7;
    A(2, 2) = 9;
    vec b{1.0, 2.0, 3.0};

    auto f = lu(assume_square(A));
    EXPECT_FALSE(f.singular);

    vec x(3);
    lu_solve(f, b, x);

    // Verify A*x = b
    EXPECT_NEAR(2 * x[0] + x[1], 1.0, 1e-12);
    EXPECT_NEAR(4 * x[0] + 3 * x[1] + 2 * x[2], 2.0, 1e-12);
    EXPECT_NEAR(8 * x[0] + 7 * x[1] + 9 * x[2], 3.0, 1e-12);
}

TEST(LU, SolveIdentitySystem) {
    idx n = 5;
    mat A(n, n, 0.0);
    for (idx i = 0; i < n; ++i) {
        A(i, i) = 1.0;
    }
    vec b{1.0, 2.0, 3.0, 4.0, 5.0};

    auto f = lu(assume_square(A));
    vec x(n);
    lu_solve(f, b, x);

    for (idx i = 0; i < n; ++i) {
        EXPECT_NEAR(x[i], b[i], 1e-12);
    }
}

TEST(LU, SolveDiagonalSystem) {
    // Diagonal matrix: trivial but exercises pivot logic
    mat A(4, 4, 0.0);
    A(0, 0) = 3;
    A(1, 1) = 6;
    A(2, 2) = 1;
    A(3, 3) = 4;
    vec b{3.0, 12.0, 5.0, 8.0};

    auto f = lu(assume_square(A));
    vec x(4);
    lu_solve(f, b, x);

    EXPECT_NEAR(x[0], 1.0, 1e-12);
    EXPECT_NEAR(x[1], 2.0, 1e-12);
    EXPECT_NEAR(x[2], 5.0, 1e-12);
    EXPECT_NEAR(x[3], 2.0, 1e-12);
}

TEST(LU, SolveLargerSystem) {
    // Random-ish 6x6 diagonally dominant system;
    // solution is x = [1, 1, 1, 1, 1, 1] by construction (b = A * ones)
    idx n = 6;
    mat A(n, n, 0.0);
    // tridiagonal + dominant diagonal
    for (idx i = 0; i < n; ++i) {
        A(i, i) = 10.0;
        if (i > 0) {
            A(i, i - 1) = -1.0;
        }
        if (i < n - 1) {
            A(i, i + 1) = -2.0;
        }
    }
    // b = A * ones
    vec b(n, 0.0);
    for (idx i = 0; i < n; ++i) {
        for (idx j = 0; j < n; ++j) {
            b[i] += A(i, j) * 1.0;
        }
    }

    auto f = lu(assume_square(A));
    vec x(n);
    lu_solve(f, b, x);
    for (idx i = 0; i < n; ++i) {
        EXPECT_NEAR(x[i], 1.0, 1e-10);
    }
}

TEST(LU, Determinant2x2) {
    // det([3 8; 4 6]) = 18 - 32 = -14
    mat A(2, 2, 0.0);
    A(0, 0) = 3;
    A(0, 1) = 8;
    A(1, 0) = 4;
    A(1, 1) = 6;
    auto f = lu(assume_square(A));
    EXPECT_NEAR(lu_det(f), -14.0, 1e-10);
}

TEST(LU, Determinant3x3) {
    // Vandermonde [1 1 1; 2 4 8; 3 9 27]  -- det =
    // 1*(4*27-8*9)-1*(2*27-8*3)+1*(2*9-4*3) = (108-72) - (54-24) + (18-12) = 36
    // - 30 + 6 = 12
    mat A(3, 3, 0.0);
    A(0, 0) = 1;
    A(0, 1) = 1;
    A(0, 2) = 1;
    A(1, 0) = 2;
    A(1, 1) = 4;
    A(1, 2) = 8;
    A(2, 0) = 3;
    A(2, 1) = 9;
    A(2, 2) = 27;
    auto f = lu(assume_square(A));
    EXPECT_NEAR(lu_det(f), 12.0, 1e-9);
}

TEST(LU, InverseTimesOriginal) {
    // A * A^{-1} = I  (to machine precision)
    mat A(3, 3, 0.0);
    A(0, 0) = 2;
    A(0, 1) = 1;
    A(0, 2) = 0;
    A(1, 0) = 1;
    A(1, 1) = 3;
    A(1, 2) = 1;
    A(2, 0) = 0;
    A(2, 1) = 1;
    A(2, 2) = 2;

    auto f = lu(assume_square(A));
    mat Ainv = lu_inv(f);

    // Check A * Ainv ~= I
    for (idx i = 0; i < 3; ++i) {
        for (idx j = 0; j < 3; ++j) {
            real entry = 0;
            for (idx k = 0; k < 3; ++k) {
                entry += A(i, k) * Ainv(k, j);
            }
            real expected = (i == j) ? 1.0 : 0.0;
            EXPECT_NEAR(entry, expected, 1e-12);
        }
    }
}

TEST(LU, MultipleRHS) {
    // Solve A X = B where B has 2 columns
    mat A(3, 3, 0.0);
    A(0, 0) = 4;
    A(0, 1) = 1;
    A(0, 2) = 0;
    A(1, 0) = 1;
    A(1, 1) = 4;
    A(1, 2) = 1;
    A(2, 0) = 0;
    A(2, 1) = 1;
    A(2, 2) = 4;

    mat B(3, 2, 0.0);
    B(0, 0) = 1;
    B(1, 0) = 0;
    B(2, 0) = 0; // first column: e_0
    B(0, 1) = 0;
    B(1, 1) = 1;
    B(2, 1) = 0; // second column: e_1

    for (const auto &factor : {seq::lu(A), lapack::lu(A)}) {
        mat X;
        lu_solve(factor, B, X);

        for (idx column = 0; column < B.cols(); ++column) {
            for (idx row = 0; row < B.rows(); ++row) {
                real value = 0.0;
                for (idx k = 0; k < A.cols(); ++k) {
                    value += A(row, k) * X(k, column);
                }
                EXPECT_NEAR(value, B(row, column), 1e-12);
            }
        }
    }
}

TEST(LU, SingularMatrix) {
    mat A(3, 3, 0.0);
    // Rank-1: all rows are [1, 2, 3]
    A(0, 0) = 1;
    A(0, 1) = 2;
    A(0, 2) = 3;
    A(1, 0) = 1;
    A(1, 1) = 2;
    A(1, 2) = 3;
    A(2, 0) = 1;
    A(2, 1) = 2;
    A(2, 2) = 3;
    auto f = lu(assume_square(A));
    EXPECT_TRUE(f.singular);
}

// QR factorization

// Helper: check ||Q^T Q - I||_inf < tol
static void expect_orthogonal(const mat &Q, real tol = 1e-10) {
    const idx m = Q.rows();
    for (idx i = 0; i < m; ++i) {
        for (idx j = 0; j < m; ++j) {
            real entry = 0;
            for (idx k = 0; k < m; ++k) {
                entry += Q(k, i) * Q(k, j);
            }
            real expected = (i == j) ? 1.0 : 0.0;
            EXPECT_NEAR(entry, expected, tol) << "Q^T Q [" << i << "," << j << "]";
        }
    }
}

// Helper: check ||Q R - A||_inf < tol
static void expect_qr_product(const mat &Q, const mat &R, const mat &A, real tol = 1e-10) {
    for (idx i = 0; i < A.rows(); ++i) {
        for (idx j = 0; j < A.cols(); ++j) {
            real qr_ij = 0;
            for (idx k = 0; k < Q.cols(); ++k) {
                qr_ij += Q(i, k) * R(k, j);
            }
            EXPECT_NEAR(qr_ij, A(i, j), tol) << "QR [" << i << "," << j << "]";
        }
    }
}

TEST(QR, OrthogonalitySquare3x3) {
    mat A(3, 3, 0.0);
    A(0, 0) = 12;
    A(0, 1) = -51;
    A(0, 2) = 4;
    A(1, 0) = 6;
    A(1, 1) = 167;
    A(1, 2) = -68;
    A(2, 0) = -4;
    A(2, 1) = 24;
    A(2, 2) = -41;

    auto f = qr(A);
    expect_orthogonal(f.Q);
}

TEST(QR, ProductRecoversA_3x3) {
    mat A(3, 3, 0.0);
    A(0, 0) = 12;
    A(0, 1) = -51;
    A(0, 2) = 4;
    A(1, 0) = 6;
    A(1, 1) = 167;
    A(1, 2) = -68;
    A(2, 0) = -4;
    A(2, 1) = 24;
    A(2, 2) = -41;

    auto f = qr(A);
    expect_qr_product(f.Q, f.R, A);
}

TEST(QR, RIsUpperTriangular) {
    mat A(4, 3, 0.0);
    A(0, 0) = 1;
    A(0, 1) = 2;
    A(0, 2) = 3;
    A(1, 0) = 4;
    A(1, 1) = 5;
    A(1, 2) = 6;
    A(2, 0) = 7;
    A(2, 1) = 8;
    A(2, 2) = 10;
    A(3, 0) = 0;
    A(3, 1) = 1;
    A(3, 2) = 2;

    auto f = qr(A);
    // All sub-diagonal entries of R must be zero
    for (idx i = 1; i < f.R.rows(); ++i) {
        for (idx j = 0; j < std::min(i, f.R.cols()); ++j) {
            EXPECT_NEAR(f.R(i, j), 0.0, 1e-10);
        }
    }
}

TEST(QR, ProductRecoversA_Overdetermined) {
    // 4x3 overdetermined system
    mat A(4, 3, 0.0);
    A(0, 0) = 1;
    A(0, 1) = 2;
    A(0, 2) = 3;
    A(1, 0) = 4;
    A(1, 1) = 5;
    A(1, 2) = 6;
    A(2, 0) = 7;
    A(2, 1) = 8;
    A(2, 2) = 10;
    A(3, 0) = 0;
    A(3, 1) = 1;
    A(3, 2) = 2;

    auto f = qr(A);
    expect_orthogonal(f.Q);
    expect_qr_product(f.Q, f.R, A);
}

TEST(QR, SolveSquareExact) {
    // A = [[2,1,0],[1,3,1],[0,1,2]]
    // x_true = [2, 3, 2]
    // b = A * x_true = [2*2+1*3, 1*2+3*3+1*2, 1*3+2*2] = [7, 13, 7]
    mat A(3, 3, 0.0);
    A(0, 0) = 2;
    A(0, 1) = 1;
    A(1, 0) = 1;
    A(1, 1) = 3;
    A(1, 2) = 1;
    A(2, 1) = 1;
    A(2, 2) = 2;
    vec b{7.0, 13.0, 7.0};

    auto f = qr(A);
    vec x(3);
    qr_solve(f, b, x);

    EXPECT_NEAR(x[0], 2.0, 1e-10);
    EXPECT_NEAR(x[1], 3.0, 1e-10);
    EXPECT_NEAR(x[2], 2.0, 1e-10);
}

TEST(QR, SolveLeastSquares) {
    // Overdetermined: fit y = a + b*t to 4 data points
    // t = [0, 1, 2, 3],  y = [1, 2, 3, 3.5]  (nearly linear)
    // A = [1 0; 1 1; 1 2; 1 3],  b = [1; 2; 3; 3.5]
    mat A(4, 2, 0.0);
    A(0, 0) = 1;
    A(0, 1) = 0;
    A(1, 0) = 1;
    A(1, 1) = 1;
    A(2, 0) = 1;
    A(2, 1) = 2;
    A(3, 0) = 1;
    A(3, 1) = 3;
    vec b{1.0, 2.0, 3.0, 3.5};

    auto f = qr(A);
    vec x(2);
    qr_solve(f, b, x);

    // Verify the normal equations A^T A x = A^T b hold
    // A^T A = [[4, 6],[6, 14]],  A^T b = [9.5, 18.5]
    // Solution: x = [0.95, 0.85] (from normal equations)
    // Check residual ||A*x - b||^2 is minimised: any perturbation makes it
    // larger
    real res = 0;
    for (idx i = 0; i < 4; ++i) {
        real ri = (A(i, 0) * x[0]) + (A(i, 1) * x[1]) - b[i];
        res += ri * ri;
    }

    // Perturb x slightly and verify residual increases
    vec x1 = x;
    x1[0] += 0.1;
    real res1 = 0;
    for (idx i = 0; i < 4; ++i) {
        real ri = (A(i, 0) * x1[0]) + (A(i, 1) * x1[1]) - b[i];
        res1 += ri * ri;
    }
    EXPECT_LT(res, res1);
}

TEST(QR, IdentityMatrix) {
    // Householder QR on I produces R = diag(+/-1,+/-1,...) not necessarily +1.
    // Each reflector has det = -1, so signs can flip.
    // Correctness check: Q is orthogonal and Q*R = I.
    idx n = 4;
    mat A(n, n, 0.0);
    for (idx i = 0; i < n; ++i) {
        A(i, i) = 1.0;
    }
    auto f = qr(A);
    expect_orthogonal(f.Q);
    expect_qr_product(f.Q, f.R, A);
    // R diagonal must be +/-1
    for (idx i = 0; i < n; ++i) {
        EXPECT_NEAR(std::abs(f.R(i, i)), 1.0, 1e-10);
    }
}

TEST(Cholesky, SPDSolve) {
    mat A(3, 3, 0.0);
    A(0, 0) = 4.0;
    A(0, 1) = 1.0;
    A(1, 0) = 1.0;
    A(1, 1) = 3.0;
    A(1, 2) = 1.0;
    A(2, 1) = 1.0;
    A(2, 2) = 2.0;

    vec b{1.0, 2.0, 3.0};
    vec x(3, 0.0);
    auto f = cholesky(linear::assume_spd(A));
    ASSERT_TRUE(f.success);
    cholesky_solve(f, b, x);

    vec Ax(3);
    matvec(A, x, Ax);
    for (idx i = 0; i < 3; ++i) {
        EXPECT_NEAR(Ax[i], b[i], 1e-10);
    }
}

TEST(Cholesky, MultipleRHS) {
    mat A(3, 3, 0.0);
    A(0, 0) = 4.0;
    A(0, 1) = A(1, 0) = 1.0;
    A(1, 1) = 3.0;
    A(1, 2) = A(2, 1) = 1.0;
    A(2, 2) = 2.0;

    mat B(3, 2, 0.0);
    B(0, 0) = 1.0;
    B(1, 1) = 1.0;
    mat X;
    const auto factor = cholesky(linear::assume_spd(A));
    ASSERT_TRUE(factor.success);
    cholesky_solve(factor, B, X);

    mat product(3, 2, 0.0);
    matmul(A, X, product);
    for (idx row = 0; row < B.rows(); ++row) {
        for (idx column = 0; column < B.cols(); ++column) {
            EXPECT_NEAR(product(row, column), B(row, column), 1e-12);
        }
    }
}

TEST(Cholesky, IndefiniteFails) {
    mat A(2, 2, 0.0);
    A(0, 0) = 1.0;
    A(1, 1) = -1.0;

    // Deliberately indefinite: assume_spd would reject it before the factorization
    // ever runs, which is the point of the opt-out.
    auto f = unsafe::cholesky(A);
    EXPECT_FALSE(f.success);
}
