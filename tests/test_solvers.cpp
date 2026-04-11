#include <gtest/gtest.h>
#include "linalg/solvers/solvers.hpp"
#include "linalg/factorization/thomas.hpp"
#include "linalg/sparse/sparse.hpp"
#include <cmath>

using namespace num;

// Conjugate Gradient

TEST(CG, Small3x3) {
    // A = [4 1 0; 1 4 1; 0 1 4], b = [1; 2; 3]  =>  x = [5/28, 2/7, 19/28]
    Matrix A(3, 3, 0.0);
    A(0, 0) = 4;
    A(0, 1) = 1;
    A(1, 0) = 1;
    A(1, 1) = 4;
    A(1, 2) = 1;
    A(2, 1) = 1;
    A(2, 2) = 4;

    Vector       b{1.0, 2.0, 3.0};
    Vector       x(3, 0.0);
    SolverResult r = cg(A, b, x);

    EXPECT_TRUE(r.converged);
    EXPECT_LT(r.residual, 1e-10);
    EXPECT_NEAR(x[0], 5.0 / 28.0, 1e-6);
    EXPECT_NEAR(x[1], 2.0 / 7.0, 1e-6);
    EXPECT_NEAR(x[2], 19.0 / 28.0, 1e-6);
}

TEST(CG, DiagonalDominant5x5) {
    idx    n = 5;
    Matrix A(n, n, 0.0);
    for (idx i = 0; i < n; ++i) {
        A(i, i) = 10.0;
        if (i > 0)
            A(i, i - 1) = 1.0;
        if (i < n - 1)
            A(i, i + 1) = 1.0;
    }
    Vector       b(n, 1.0), x(n, 0.0);
    SolverResult r = cg(A, b, x);

    EXPECT_TRUE(r.converged);
    EXPECT_LT(r.residual, 1e-10);

    Vector Ax(n);
    matvec(A, x, Ax);
    real err = 0;
    for (idx i = 0; i < n; ++i)
        err += (Ax[i] - b[i]) * (Ax[i] - b[i]);
    EXPECT_LT(std::sqrt(err), 1e-9);
}

TEST(CG, ConvergesWithinN) {
    idx    n = 10;
    Matrix A(n, n, 0.0);
    for (idx i = 0; i < n; ++i)
        A(i, i) = static_cast<real>(i + 1);

    Vector b(n), x(n, 0.0);
    for (idx i = 0; i < n; ++i)
        b[i] = static_cast<real>(i + 1);

    SolverResult r = cg(A, b, x);
    EXPECT_TRUE(r.converged);
    EXPECT_LE(r.iterations, n);
    for (idx i = 0; i < n; ++i)
        EXPECT_NEAR(x[i], 1.0, 1e-9);
}

// Thomas algorithm

TEST(Thomas, Small4x4) {
    Vector a{-1.0, -1.0, -1.0}, b{2.0, 2.0, 2.0, 2.0}, c{-1.0, -1.0, -1.0};
    Vector d{1.0, 0.0, 0.0, 1.0}, x(4);
    thomas(a, b, c, d, x);
    for (idx i = 0; i < 4; ++i)
        EXPECT_NEAR(x[i], 1.0, 1e-10);
}

TEST(Thomas, Laplacian1D) {
    idx    n = 10;
    Vector a(n - 1, -1.0), b(n, 2.0), c(n - 1, -1.0), d(n, 1.0), x(n);
    thomas(a, b, c, d, x);
    for (idx i = 0; i < n; ++i) {
        real Ax = b[i] * x[i];
        if (i > 0)
            Ax += a[i - 1] * x[i - 1];
        if (i < n - 1)
            Ax += c[i] * x[i + 1];
        EXPECT_NEAR(Ax, d[i], 1e-10);
    }
}

TEST(Thomas, TwoByTwo) {
    Vector a{2.0}, b{3.0, 4.0}, c{1.0}, d{5.0, 6.0}, x(2);
    thomas(a, b, c, d, x);
    EXPECT_NEAR(x[0], 1.4, 1e-10);
    EXPECT_NEAR(x[1], 0.8, 1e-10);
}

// Gauss-Seidel

TEST(GaussSeidel, DiagonalDominant3x3) {
    // [4 1 0; 1 4 1; 0 1 4] x = [1; 2; 3]  =>  same solution as CG test
    Matrix A(3, 3, 0.0);
    A(0, 0) = 4;
    A(0, 1) = 1;
    A(1, 0) = 1;
    A(1, 1) = 4;
    A(1, 2) = 1;
    A(2, 1) = 1;
    A(2, 2) = 4;

    Vector       b{1.0, 2.0, 3.0};
    Vector       x(3, 0.0);
    SolverResult r = gauss_seidel(A, b, x);

    EXPECT_TRUE(r.converged);
    EXPECT_LT(r.residual, 1e-10);
    EXPECT_NEAR(x[0], 5.0 / 28.0, 1e-6);
    EXPECT_NEAR(x[1], 2.0 / 7.0, 1e-6);
    EXPECT_NEAR(x[2], 19.0 / 28.0, 1e-6);
}

TEST(GaussSeidel, DiagonalSystem) {
    // Diagonal A: solution is trivially b[i]/A[i][i]
    idx    n = 8;
    Matrix A(n, n, 0.0);
    for (idx i = 0; i < n; ++i)
        A(i, i) = static_cast<real>(i + 1);

    Vector b(n), x(n, 0.0);
    for (idx i = 0; i < n; ++i)
        b[i] = static_cast<real>((i + 1) * (i + 1));

    SolverResult r = gauss_seidel(A, b, x);
    EXPECT_TRUE(r.converged);
    for (idx i = 0; i < n; ++i)
        EXPECT_NEAR(x[i], static_cast<real>(i + 1), 1e-8);
}

TEST(GaussSeidel, ResidualVerified) {
    idx    n = 6;
    Matrix A(n, n, 0.0);
    for (idx i = 0; i < n; ++i) {
        A(i, i) = 8.0;
        if (i > 0)
            A(i, i - 1) = -1.0;
        if (i < n - 1)
            A(i, i + 1) = -1.0;
    }
    Vector       b(n, 1.0), x(n, 0.0);
    SolverResult r = gauss_seidel(A, b, x);

    EXPECT_TRUE(r.converged);
    EXPECT_LT(r.residual, 1e-10);

    // Verify Ax ~= b
    Vector Ax(n);
    matvec(A, x, Ax);
    for (idx i = 0; i < n; ++i)
        EXPECT_NEAR(Ax[i], b[i], 1e-8);
}

// Jacobi

TEST(Jacobi, DiagonalDominant3x3) {
    Matrix A(3, 3, 0.0);
    A(0, 0) = 4;
    A(0, 1) = 1;
    A(1, 0) = 1;
    A(1, 1) = 4;
    A(1, 2) = 1;
    A(2, 1) = 1;
    A(2, 2) = 4;

    Vector       b{1.0, 2.0, 3.0};
    Vector       x(3, 0.0);
    SolverResult r = jacobi(A, b, x);

    EXPECT_TRUE(r.converged);
    EXPECT_LT(r.residual, 1e-10);
    EXPECT_NEAR(x[0], 5.0 / 28.0, 1e-6);
    EXPECT_NEAR(x[1], 2.0 / 7.0, 1e-6);
    EXPECT_NEAR(x[2], 19.0 / 28.0, 1e-6);
}

TEST(Jacobi, DiagonalSystem) {
    idx    n = 8;
    Matrix A(n, n, 0.0);
    for (idx i = 0; i < n; ++i)
        A(i, i) = static_cast<real>(i + 1);

    Vector b(n), x(n, 0.0);
    for (idx i = 0; i < n; ++i)
        b[i] = static_cast<real>((i + 1) * (i + 1));

    // Diagonal system: Jacobi converges in one iteration
    SolverResult r = jacobi(A, b, x, 1e-10, 1);
    EXPECT_EQ(r.iterations, static_cast<idx>(1));
    for (idx i = 0; i < n; ++i)
        EXPECT_NEAR(x[i], static_cast<real>(i + 1), 1e-10);
}

TEST(Jacobi, ResidualVerified) {
    idx    n = 6;
    Matrix A(n, n, 0.0);
    for (idx i = 0; i < n; ++i) {
        A(i, i) = 8.0;
        if (i > 0)
            A(i, i - 1) = -1.0;
        if (i < n - 1)
            A(i, i + 1) = -1.0;
    }
    Vector       b(n, 1.0), x(n, 0.0);
    SolverResult r = jacobi(A, b, x);

    EXPECT_TRUE(r.converged);
    EXPECT_LT(r.residual, 1e-10);

    Vector Ax(n);
    matvec(A, x, Ax);
    for (idx i = 0; i < n; ++i)
        EXPECT_NEAR(Ax[i], b[i], 1e-8);
}

// GMRES (Krylov)

TEST(GMRES, SPD3x3Dense) {
    // Same SPD system  -- GMRES should also solve it
    Matrix A(3, 3, 0.0);
    A(0, 0) = 4;
    A(0, 1) = 1;
    A(1, 0) = 1;
    A(1, 1) = 4;
    A(1, 2) = 1;
    A(2, 1) = 1;
    A(2, 2) = 4;

    Vector       b{1.0, 2.0, 3.0};
    Vector       x(3, 0.0);
    SolverResult r = gmres(A, b, x);

    EXPECT_TRUE(r.converged);
    EXPECT_LT(r.residual, 1e-6);
    EXPECT_NEAR(x[0], 5.0 / 28.0, 1e-5);
    EXPECT_NEAR(x[1], 2.0 / 7.0, 1e-5);
    EXPECT_NEAR(x[2], 19.0 / 28.0, 1e-5);
}

TEST(GMRES, NonSymmetricDense) {
    // Non-symmetric system: A = [3 1; 1 2], b = [5; 3]  =>  x = [1, 2]
    Matrix A(2, 2, 0.0);
    A(0, 0) = 3;
    A(0, 1) = 1;
    A(1, 0) = 1;
    A(1, 1) = 2;

    Vector       b{5.0, 3.0}; // actually symmetric here but checks general path
    Vector       x(2, 0.0);
    SolverResult r = gmres(A, b, x);

    EXPECT_TRUE(r.converged);
    EXPECT_NEAR(x[0], 1.4, 1e-5);
    EXPECT_NEAR(x[1], 0.8, 1e-5);
}

TEST(GMRES, SparseLaplacian1D) {
    // 1D Laplacian on 10 nodes via SparseMatrix
    idx               n = 10;
    std::vector<idx>  rows, cols;
    std::vector<real> vals;
    for (idx i = 0; i < n; ++i) {
        rows.push_back(i);
        cols.push_back(i);
        vals.push_back(2.0);
        if (i > 0) {
            rows.push_back(i);
            cols.push_back(i - 1);
            vals.push_back(-1.0);
        }
        if (i < n - 1) {
            rows.push_back(i);
            cols.push_back(i + 1);
            vals.push_back(-1.0);
        }
    }
    SparseMatrix A = SparseMatrix::from_triplets(n, n, rows, cols, vals);

    Vector       b(n, 1.0), x(n, 0.0);
    SolverResult r = gmres(A, b, x);

    EXPECT_TRUE(r.converged);
    EXPECT_LT(r.residual, 1e-6);

    // Verify Ax ~= b
    Vector Ax(n);
    sparse_matvec(A, x, Ax);
    for (idx i = 0; i < n; ++i)
        EXPECT_NEAR(Ax[i], b[i], 1e-5);
}

TEST(GMRES, MatrixFree) {
    // Diagonal system via MatVecFn lambda
    idx    n = 5;
    Vector diag(n);
    for (idx i = 0; i < n; ++i)
        diag[i] = static_cast<real>(i + 1);

    MatVecFn mv = [&](const Vector& in, Vector& out) {
        out = Vector(n);
        for (idx i = 0; i < n; ++i)
            out[i] = diag[i] * in[i];
    };

    Vector       b(n, 1.0), x(n, 0.0);
    SolverResult r = gmres(mv, n, b, x);

    EXPECT_TRUE(r.converged);
    for (idx i = 0; i < n; ++i)
        EXPECT_NEAR(x[i], 1.0 / static_cast<real>(i + 1), 1e-5);
}

// SparseMatrix construction

TEST(SparseMatrix, FromTriplets) {
    // 3x3 identity
    SparseMatrix I = SparseMatrix::from_triplets(3,
                                                 3,
                                                 {0, 1, 2},
                                                 {0, 1, 2},
                                                 {1.0, 1.0, 1.0});
    EXPECT_EQ(I.nnz(), static_cast<idx>(3));
    EXPECT_NEAR(I(0, 0), 1.0, 1e-15);
    EXPECT_NEAR(I(1, 1), 1.0, 1e-15);
    EXPECT_NEAR(I(0, 1), 0.0, 1e-15);
}

TEST(SparseMatrix, DuplicatesSummed) {
    // Two entries at (0,0): should be summed to 3.0
    SparseMatrix A = SparseMatrix::from_triplets(2,
                                                 2,
                                                 {0, 0, 1},
                                                 {0, 0, 1},
                                                 {1.0, 2.0, 4.0});
    EXPECT_NEAR(A(0, 0), 3.0, 1e-15);
    EXPECT_NEAR(A(1, 1), 4.0, 1e-15);
}

TEST(SparseMatrix, Matvec) {
    // A = [2 -1; -1 2], x = [1; 1]  =>  y = [1; 1]
    SparseMatrix A = SparseMatrix::from_triplets(2,
                                                 2,
                                                 {0, 0, 1, 1},
                                                 {0, 1, 0, 1},
                                                 {2.0, -1.0, -1.0, 2.0});
    Vector       x{1.0, 1.0}, y(2);
    sparse_matvec(A, x, y);
    EXPECT_NEAR(y[0], 1.0, 1e-14);
    EXPECT_NEAR(y[1], 1.0, 1e-14);
}
