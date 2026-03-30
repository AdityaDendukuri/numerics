#include <gtest/gtest.h>
#include "core/matrix.hpp"

using namespace num;

TEST(Matrix, Construction) {
    Matrix m(3, 4);
    EXPECT_EQ(m.rows(), 3);
    EXPECT_EQ(m.cols(), 4);
    EXPECT_EQ(m.size(), 12);

    Matrix m2(2, 2, 5.0);
    EXPECT_DOUBLE_EQ(m2(0, 0), 5.0);
    EXPECT_DOUBLE_EQ(m2(1, 1), 5.0);
}

TEST(Matrix, CopyMove) {
    Matrix m(2, 2, 1.0);
    m(0, 1) = 2.0;

    Matrix copy = m;
    EXPECT_DOUBLE_EQ(copy(0, 1), 2.0);

    Matrix moved = std::move(copy);
    EXPECT_EQ(moved.rows(), 2);
}

TEST(Matrix, Matvec) {
    Matrix A(2, 3);
    A(0, 0) = 1; A(0, 1) = 2; A(0, 2) = 3;
    A(1, 0) = 4; A(1, 1) = 5; A(1, 2) = 6;

    Vector x{1.0, 1.0, 1.0};
    Vector y(2);

    matvec(A, x, y);
    EXPECT_DOUBLE_EQ(y[0], 6.0);
    EXPECT_DOUBLE_EQ(y[1], 15.0);
}

TEST(Matrix, Matmul) {
    Matrix A(2, 3);
    A(0, 0) = 1; A(0, 1) = 2; A(0, 2) = 3;
    A(1, 0) = 4; A(1, 1) = 5; A(1, 2) = 6;

    Matrix B(3, 2);
    B(0, 0) = 1; B(0, 1) = 2;
    B(1, 0) = 3; B(1, 1) = 4;
    B(2, 0) = 5; B(2, 1) = 6;

    Matrix C(2, 2);
    matmul(A, B, C);

    EXPECT_DOUBLE_EQ(C(0, 0), 22.0);
    EXPECT_DOUBLE_EQ(C(0, 1), 28.0);
    EXPECT_DOUBLE_EQ(C(1, 0), 49.0);
    EXPECT_DOUBLE_EQ(C(1, 1), 64.0);
}

TEST(Matrix, Matadd) {
    Matrix A(2, 2, 1.0);
    Matrix B(2, 2, 2.0);
    Matrix C(2, 2);

    matadd(2.0, A, 3.0, B, C);
    EXPECT_DOUBLE_EQ(C(0, 0), 8.0);
}

// Backend correctness: every backend must produce the same result as Backend::seq

static Matrix make_test_matrix(idx rows, idx cols) {
    Matrix A(rows, cols);
    for (idx i = 0; i < rows; ++i)
        for (idx j = 0; j < cols; ++j)
            A(i, j) = static_cast<real>(i * cols + j + 1);
    return A;
}

static Vector make_test_vector(idx n) {
    Vector v(n);
    for (idx i = 0; i < n; ++i)
        v[i] = static_cast<real>(i + 1);
    return v;
}

TEST(MatmulPolicy, BlockedMatchesSeq) {
    Matrix A = make_test_matrix(32, 32);
    Matrix B = make_test_matrix(32, 32);
    Matrix C_seq(32, 32), C_blk(32, 32);
    matmul(A, B, C_seq, Backend::seq);
    matmul(A, B, C_blk, Backend::blocked);
    for (idx i = 0; i < 32; ++i)
        for (idx j = 0; j < 32; ++j)
            EXPECT_NEAR(C_blk(i, j), C_seq(i, j), 1e-10);
}

TEST(MatvecPolicy, BlockedMatchesSeq) {
    Matrix A = make_test_matrix(16, 16);
    Vector x = make_test_vector(16);
    Vector y_seq(16), y_blk(16);
    matvec(A, x, y_seq, Backend::seq);
    matvec(A, x, y_blk, Backend::blocked);
    for (idx i = 0; i < 16; ++i)
        EXPECT_NEAR(y_blk[i], y_seq[i], 1e-10);
}
