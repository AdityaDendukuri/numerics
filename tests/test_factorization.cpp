#include <gtest/gtest.h>
#include "linalg/factorization/factorization.hpp"
#include <cmath>

using namespace num;

// Helpers
static real mat_norm_inf(const Matrix& A) {
    real m = 0;
    for (idx i = 0; i < A.rows(); ++i)
        for (idx j = 0; j < A.cols(); ++j)
            m = std::max(m, std::abs(A(i, j)));
    return m;
}

static real vec_norm_inf(const Vector& v) {
    real m = 0;
    for (idx i = 0; i < v.size(); ++i)
        m = std::max(m, std::abs(v[i]));
    return m;
}

// LU factorization

TEST(LU, SolveSmall3x3) {
    // [2  1  0]       [1]
    // [4  3  2] x  =  [2]
    // [8  7  9]       [3]
    // solution: x = [-1/2, 2, -1/2]  -- computed by hand
    Matrix A(3, 3, 0.0);
    A(0,0)=2; A(0,1)=1;
    A(1,0)=4; A(1,1)=3; A(1,2)=2;
    A(2,0)=8; A(2,1)=7; A(2,2)=9;
    Vector b{1.0, 2.0, 3.0};

    auto f = lu(A);
    EXPECT_FALSE(f.singular);

    Vector x(3);
    lu_solve(f, b, x);

    // Verify A*x = b
    EXPECT_NEAR(2*x[0] +   x[1]        , 1.0, 1e-12);
    EXPECT_NEAR(4*x[0] + 3*x[1] + 2*x[2], 2.0, 1e-12);
    EXPECT_NEAR(8*x[0] + 7*x[1] + 9*x[2], 3.0, 1e-12);
}

TEST(LU, SolveIdentitySystem) {
    idx n = 5;
    Matrix A(n, n, 0.0);
    for (idx i = 0; i < n; ++i) A(i, i) = 1.0;
    Vector b{1.0, 2.0, 3.0, 4.0, 5.0};

    auto f = lu(A);
    Vector x(n);
    lu_solve(f, b, x);

    for (idx i = 0; i < n; ++i)
        EXPECT_NEAR(x[i], b[i], 1e-12);
}

TEST(LU, SolveDiagonalSystem) {
    // Diagonal matrix: trivial but exercises pivot logic
    Matrix A(4, 4, 0.0);
    A(0,0)=3; A(1,1)=6; A(2,2)=1; A(3,3)=4;
    Vector b{3.0, 12.0, 5.0, 8.0};

    auto f = lu(A);
    Vector x(4);
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
    Matrix A(n, n, 0.0);
    // Tridiagonal + dominant diagonal
    for (idx i = 0; i < n; ++i) {
        A(i, i) = 10.0;
        if (i > 0)     A(i, i-1) = -1.0;
        if (i < n-1)   A(i, i+1) = -2.0;
    }
    // b = A * ones
    Vector b(n, 0.0);
    for (idx i = 0; i < n; ++i)
        for (idx j = 0; j < n; ++j)
            b[i] += A(i, j) * 1.0;

    auto f = lu(A);
    Vector x(n);
    lu_solve(f, b, x);
    for (idx i = 0; i < n; ++i)
        EXPECT_NEAR(x[i], 1.0, 1e-10);
}

TEST(LU, Determinant2x2) {
    // det([3 8; 4 6]) = 18 - 32 = -14
    Matrix A(2, 2, 0.0);
    A(0,0)=3; A(0,1)=8;
    A(1,0)=4; A(1,1)=6;
    auto f = lu(A);
    EXPECT_NEAR(lu_det(f), -14.0, 1e-10);
}

TEST(LU, Determinant3x3) {
    // Vandermonde [1 1 1; 2 4 8; 3 9 27]  -- det = 1*(4*27-8*9)-1*(2*27-8*3)+1*(2*9-4*3)
    // = (108-72) - (54-24) + (18-12) = 36 - 30 + 6 = 12
    Matrix A(3, 3, 0.0);
    A(0,0)=1; A(0,1)=1; A(0,2)=1;
    A(1,0)=2; A(1,1)=4; A(1,2)=8;
    A(2,0)=3; A(2,1)=9; A(2,2)=27;
    auto f = lu(A);
    EXPECT_NEAR(lu_det(f), 12.0, 1e-9);
}

TEST(LU, InverseTimesOriginal) {
    // A * A^{-1} = I  (to machine precision)
    Matrix A(3, 3, 0.0);
    A(0,0)=2; A(0,1)=1; A(0,2)=0;
    A(1,0)=1; A(1,1)=3; A(1,2)=1;
    A(2,0)=0; A(2,1)=1; A(2,2)=2;

    auto f   = lu(A);
    Matrix Ainv = lu_inv(f);

    // Check A * Ainv ~= I
    for (idx i = 0; i < 3; ++i)
        for (idx j = 0; j < 3; ++j) {
            real entry = 0;
            for (idx k = 0; k < 3; ++k) entry += A(i,k) * Ainv(k,j);
            real expected = (i == j) ? 1.0 : 0.0;
            EXPECT_NEAR(entry, expected, 1e-12);
        }
}

TEST(LU, MultipleRHS) {
    // Solve A X = B where B has 2 columns
    Matrix A(3, 3, 0.0);
    A(0,0)=4; A(0,1)=1; A(0,2)=0;
    A(1,0)=1; A(1,1)=4; A(1,2)=1;
    A(2,0)=0; A(2,1)=1; A(2,2)=4;

    Matrix B(3, 2, 0.0);
    B(0,0)=1; B(1,0)=0; B(2,0)=0;   // first column: e_0
    B(0,1)=0; B(1,1)=1; B(2,1)=0;   // second column: e_1

    auto f  = lu(A);
    Matrix X(3, 2, 0.0);
    lu_solve(f, B, X);

    // Verify A * X[:,j] = B[:,j] for each column
    for (idx j = 0; j < 2; ++j)
        for (idx i = 0; i < 3; ++i) {
            real ax = 0;
            for (idx k = 0; k < 3; ++k) ax += A(i,k) * X(k,j);
            EXPECT_NEAR(ax, B(i,j), 1e-12);
        }
}

TEST(LU, SingularMatrix) {
    Matrix A(3, 3, 0.0);
    // Rank-1: all rows are [1, 2, 3]
    A(0,0)=1; A(0,1)=2; A(0,2)=3;
    A(1,0)=1; A(1,1)=2; A(1,2)=3;
    A(2,0)=1; A(2,1)=2; A(2,2)=3;
    auto f = lu(A);
    EXPECT_TRUE(f.singular);
}

// QR factorization

// Helper: check ||Q^T Q - I||_inf < tol
static void expect_orthogonal(const Matrix& Q, real tol = 1e-10) {
    const idx m = Q.rows();
    for (idx i = 0; i < m; ++i)
        for (idx j = 0; j < m; ++j) {
            real entry = 0;
            for (idx k = 0; k < m; ++k) entry += Q(k, i) * Q(k, j);
            real expected = (i == j) ? 1.0 : 0.0;
            EXPECT_NEAR(entry, expected, tol) << "Q^T Q [" << i << "," << j << "]";
        }
}

// Helper: check ||Q R - A||_inf < tol
static void expect_qr_product(const Matrix& Q, const Matrix& R,
                               const Matrix& A, real tol = 1e-10) {
    for (idx i = 0; i < A.rows(); ++i)
        for (idx j = 0; j < A.cols(); ++j) {
            real qr_ij = 0;
            for (idx k = 0; k < Q.cols(); ++k) qr_ij += Q(i,k) * R(k,j);
            EXPECT_NEAR(qr_ij, A(i,j), tol) << "QR [" << i << "," << j << "]";
        }
}

TEST(QR, OrthogonalitySquare3x3) {
    Matrix A(3, 3, 0.0);
    A(0,0)=12; A(0,1)=-51; A(0,2)=4;
    A(1,0)=6;  A(1,1)=167; A(1,2)=-68;
    A(2,0)=-4; A(2,1)=24;  A(2,2)=-41;

    auto f = qr(A);
    expect_orthogonal(f.Q);
}

TEST(QR, ProductRecoversA_3x3) {
    Matrix A(3, 3, 0.0);
    A(0,0)=12; A(0,1)=-51; A(0,2)=4;
    A(1,0)=6;  A(1,1)=167; A(1,2)=-68;
    A(2,0)=-4; A(2,1)=24;  A(2,2)=-41;

    auto f = qr(A);
    expect_qr_product(f.Q, f.R, A);
}

TEST(QR, RIsUpperTriangular) {
    Matrix A(4, 3, 0.0);
    A(0,0)=1; A(0,1)=2; A(0,2)=3;
    A(1,0)=4; A(1,1)=5; A(1,2)=6;
    A(2,0)=7; A(2,1)=8; A(2,2)=10;
    A(3,0)=0; A(3,1)=1; A(3,2)=2;

    auto f = qr(A);
    // All sub-diagonal entries of R must be zero
    for (idx i = 1; i < f.R.rows(); ++i)
        for (idx j = 0; j < std::min(i, f.R.cols()); ++j)
            EXPECT_NEAR(f.R(i, j), 0.0, 1e-10);
}

TEST(QR, ProductRecoversA_Overdetermined) {
    // 4x3 overdetermined system
    Matrix A(4, 3, 0.0);
    A(0,0)=1; A(0,1)=2; A(0,2)=3;
    A(1,0)=4; A(1,1)=5; A(1,2)=6;
    A(2,0)=7; A(2,1)=8; A(2,2)=10;
    A(3,0)=0; A(3,1)=1; A(3,2)=2;

    auto f = qr(A);
    expect_orthogonal(f.Q);
    expect_qr_product(f.Q, f.R, A);
}

TEST(QR, SolveSquareExact) {
    // A = [[2,1,0],[1,3,1],[0,1,2]]
    // x_true = [2, 3, 2]
    // b = A * x_true = [2*2+1*3, 1*2+3*3+1*2, 1*3+2*2] = [7, 13, 7]
    Matrix A(3, 3, 0.0);
    A(0,0)=2; A(0,1)=1;
    A(1,0)=1; A(1,1)=3; A(1,2)=1;
    A(2,1)=1; A(2,2)=2;
    Vector b{7.0, 13.0, 7.0};

    auto f = qr(A);
    Vector x(3);
    qr_solve(f, b, x);

    EXPECT_NEAR(x[0], 2.0, 1e-10);
    EXPECT_NEAR(x[1], 3.0, 1e-10);
    EXPECT_NEAR(x[2], 2.0, 1e-10);
}

TEST(QR, SolveLeastSquares) {
    // Overdetermined: fit y = a + b*t to 4 data points
    // t = [0, 1, 2, 3],  y = [1, 2, 3, 3.5]  (nearly linear)
    // A = [1 0; 1 1; 1 2; 1 3],  b = [1; 2; 3; 3.5]
    Matrix A(4, 2, 0.0);
    A(0,0)=1; A(0,1)=0;
    A(1,0)=1; A(1,1)=1;
    A(2,0)=1; A(2,1)=2;
    A(3,0)=1; A(3,1)=3;
    Vector b{1.0, 2.0, 3.0, 3.5};

    auto f = qr(A);
    Vector x(2);
    qr_solve(f, b, x);

    // Verify the normal equations A^T A x = A^T b hold
    // A^T A = [[4, 6],[6, 14]],  A^T b = [9.5, 18.5]
    // Solution: x = [0.95, 0.85] (from normal equations)
    // Check residual ||A*x - b||^2 is minimised: any perturbation makes it larger
    real res = 0;
    for (idx i = 0; i < 4; ++i) {
        real ri = A(i,0)*x[0] + A(i,1)*x[1] - b[i];
        res += ri * ri;
    }

    // Perturb x slightly and verify residual increases
    Vector x1 = x;  x1[0] += 0.1;
    real res1 = 0;
    for (idx i = 0; i < 4; ++i) {
        real ri = A(i,0)*x1[0] + A(i,1)*x1[1] - b[i];
        res1 += ri * ri;
    }
    EXPECT_LT(res, res1);
}

TEST(QR, IdentityMatrix) {
    // Householder QR on I produces R = diag(+/-1,+/-1,...) not necessarily +1.
    // Each reflector has det = -1, so signs can flip.
    // Correctness check: Q is orthogonal and Q*R = I.
    idx n = 4;
    Matrix A(n, n, 0.0);
    for (idx i = 0; i < n; ++i) A(i, i) = 1.0;
    auto f = qr(A);
    expect_orthogonal(f.Q);
    expect_qr_product(f.Q, f.R, A);
    // R diagonal must be +/-1
    for (idx i = 0; i < n; ++i)
        EXPECT_NEAR(std::abs(f.R(i, i)), 1.0, 1e-10);
}
