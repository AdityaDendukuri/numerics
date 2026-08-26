#include "container/matrix.hpp"
#include "container/matrix_expr.hpp"
#include "container/matrix_ops.hpp"
#include "linear/matrix_properties.hpp"
#include "linear/matrix_utils.hpp"
#include <gtest/gtest.h>

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
    A(0, 0) = 1;
    A(0, 1) = 2;
    A(0, 2) = 3;
    A(1, 0) = 4;
    A(1, 1) = 5;
    A(1, 2) = 6;

    Vector x{1.0, 1.0, 1.0};
    Vector y(2);

    matvec(A, x, y);
    EXPECT_DOUBLE_EQ(y[0], 6.0);
    EXPECT_DOUBLE_EQ(y[1], 15.0);
}

TEST(Matrix, Matmul) {
    Matrix A(2, 3);
    A(0, 0) = 1;
    A(0, 1) = 2;
    A(0, 2) = 3;
    A(1, 0) = 4;
    A(1, 1) = 5;
    A(1, 2) = 6;

    Matrix B(3, 2);
    B(0, 0) = 1;
    B(0, 1) = 2;
    B(1, 0) = 3;
    B(1, 1) = 4;
    B(2, 0) = 5;
    B(2, 1) = 6;

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

TEST(Matrix, RowAndElementScaling) {
    Matrix A(2, 2, 1.0);
    const std::vector<real> weights{2.0, 4.0};
    scale_rows(A, weights);
    EXPECT_DOUBLE_EQ(A(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(A(1, 0), 4.0);
    divide_rows(A, weights);
    EXPECT_DOUBLE_EQ(A(0, 1), 1.0);
    EXPECT_DOUBLE_EQ(A(1, 0), 1.0);

    Vector x{3.0, 5.0};
    scale_elements(x, weights);
    EXPECT_DOUBLE_EQ(x[0], 6.0);
    EXPECT_DOUBLE_EQ(x[1], 20.0);
    divide_elements(x, weights);
    EXPECT_DOUBLE_EQ(x[0], 3.0);
    EXPECT_DOUBLE_EQ(x[1], 5.0);
}

TEST(Matrix, RelativeSymmetryError) {
    Matrix A(2, 2, 0.0);
    A(0, 1) = 100.0;
    A(1, 0) = 100.01;
    EXPECT_NEAR(linear::relative_symmetry_error(A), 0.01 / 100.01, 1e-14);
}

// Backend correctness: every backend must produce the same result as
// backend::seq

static Matrix make_test_matrix(idx rows, idx cols) {
    Matrix A(rows, cols);
    for (idx i = 0; i < rows; ++i) {
        for (idx j = 0; j < cols; ++j) {
            A(i, j) = static_cast<real>((i * cols) + j + 1);
        }
    }
    return A;
}

static Vector make_test_vector(idx n) {
    Vector v(n);
    for (idx i = 0; i < n; ++i) {
        v[i] = static_cast<real>(i + 1);
    }
    return v;
}

TEST(MatmulPolicy, BlockedMatchesSeq) {
    Matrix A = make_test_matrix(32, 32);
    Matrix B = make_test_matrix(32, 32);
    Matrix C_seq(32, 32), C_blk(32, 32);
    matmul(A, B, C_seq, backend::seq);
    matmul(A, B, C_blk, backend::blocked);
    for (idx i = 0; i < 32; ++i) {
        for (idx j = 0; j < 32; ++j) {
            EXPECT_NEAR(C_blk(i, j), C_seq(i, j), 1e-10);
        }
    }
}

TEST(MatvecPolicy, BlockedMatchesSeq) {
    Matrix A = make_test_matrix(16, 16);
    Vector x = make_test_vector(16);
    Vector y_seq(16), y_blk(16);
    matvec(A, x, y_seq, backend::seq);
    matvec(A, x, y_blk, backend::blocked);
    for (idx i = 0; i < 16; ++i) {
        EXPECT_NEAR(y_blk[i], y_seq[i], 1e-10);
    }
}

TEST(MatrixExpr, ValueReturningMatchesOutParameterForm) {
    Matrix A(2, 3, 0.0);
    Matrix B(3, 2, 0.0);
    for (idx i = 0; i < 2; ++i) {
        for (idx j = 0; j < 3; ++j) {
            A(i, j) = 1.0 + double(i) + 2.0 * double(j);
            B(j, i) = 0.5 - double(i) + double(j);
        }
    }
    Vector x{1.0, -2.0, 3.0};

    Matrix expected_product(2, 2, 0.0);
    matmul(A, B, expected_product);
    const Matrix product = matmul(A, B);
    ASSERT_EQ(product.rows(), 2);
    ASSERT_EQ(product.cols(), 2);
    for (idx i = 0; i < 2; ++i) {
        for (idx j = 0; j < 2; ++j) {
            EXPECT_DOUBLE_EQ(product(i, j), expected_product(i, j));
        }
    }

    Vector expected_image(2, 0.0);
    matvec(A, x, expected_image);
    const Vector image = matvec(A, x);
    ASSERT_EQ(image.size(), 2);
    EXPECT_DOUBLE_EQ(image[0], expected_image[0]);
    EXPECT_DOUBLE_EQ(image[1], expected_image[1]);

    const Matrix sum = add(A, A);
    const Matrix difference = sub(A, A);
    for (idx i = 0; i < 2; ++i) {
        for (idx j = 0; j < 3; ++j) {
            EXPECT_DOUBLE_EQ(sum(i, j), 2.0 * A(i, j));
            EXPECT_DOUBLE_EQ(difference(i, j), 0.0);
        }
    }

    const Vector vector_sum = add(x, x);
    const Vector vector_difference = sub(x, x);
    for (idx i = 0; i < x.size(); ++i) {
        EXPECT_DOUBLE_EQ(vector_sum[i], 2.0 * x[i]);
        EXPECT_DOUBLE_EQ(vector_difference[i], 0.0);
    }
}

TEST(MatrixExpr, OperatorsAgreeWithNamedForms) {
    using namespace num::ops;

    Matrix A(2, 2, 0.0);
    A(0, 0) = 4.0;
    A(0, 1) = 1.0;
    A(1, 0) = 2.0;
    A(1, 1) = 3.0;
    const Vector x{1.0, 2.0};

    const Matrix product = A * A;
    const Matrix named_product = matmul(A, A);
    const Vector image = A * x;
    const Vector named_image = matvec(A, x);
    const Matrix sum = A + A;
    const Matrix difference = A - A;
    const Vector vector_sum = x + x;
    const Vector vector_difference = x - x;

    for (idx i = 0; i < 2; ++i) {
        EXPECT_DOUBLE_EQ(image[i], named_image[i]);
        EXPECT_DOUBLE_EQ(vector_sum[i], 2.0 * x[i]);
        EXPECT_DOUBLE_EQ(vector_difference[i], 0.0);
        for (idx j = 0; j < 2; ++j) {
            EXPECT_DOUBLE_EQ(product(i, j), named_product(i, j));
            EXPECT_DOUBLE_EQ(sum(i, j), 2.0 * A(i, j));
            EXPECT_DOUBLE_EQ(difference(i, j), 0.0);
        }
    }
}

TEST(MatrixExpr, NonConformingShapesThrow) {
    EXPECT_THROW((void)matmul(Matrix(2, 3, 1.0), Matrix(2, 3, 1.0)), std::invalid_argument);
    EXPECT_THROW((void)matvec(Matrix(2, 3, 1.0), Vector(2, 1.0)), std::invalid_argument);
    EXPECT_THROW((void)add(Matrix(2, 2, 1.0), Matrix(3, 3, 1.0)), std::invalid_argument);
    EXPECT_THROW((void)sub(Matrix(2, 2, 1.0), Matrix(3, 3, 1.0)), std::invalid_argument);
    EXPECT_THROW((void)add(Vector(2, 1.0), Vector(3, 1.0)), std::invalid_argument);
    EXPECT_THROW((void)sub(Vector(2, 1.0), Vector(3, 1.0)), std::invalid_argument);
}
