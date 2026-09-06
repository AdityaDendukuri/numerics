#include "container/matrix.hpp"
#include "container/matrix_expr.hpp"
#include "container/matrix_ops.hpp"
#include "linear/matrix_properties.hpp"
#include "linear/matrix_utils.hpp"
#include <gtest/gtest.h>

using namespace num;

TEST(mat, Construction) {
    mat m(3, 4);
    EXPECT_EQ(m.rows(), 3);
    EXPECT_EQ(m.cols(), 4);
    EXPECT_EQ(m.size(), 12);

    mat m2(2, 2, 5.0);
    EXPECT_DOUBLE_EQ(m2(0, 0), 5.0);
    EXPECT_DOUBLE_EQ(m2(1, 1), 5.0);
}

TEST(mat, CopyMove) {
    mat m(2, 2, 1.0);
    m(0, 1) = 2.0;

    mat copy = m;
    EXPECT_DOUBLE_EQ(copy(0, 1), 2.0);

    mat moved = std::move(copy);
    EXPECT_EQ(moved.rows(), 2);
}

TEST(mat, Matvec) {
    mat A(2, 3);
    A(0, 0) = 1;
    A(0, 1) = 2;
    A(0, 2) = 3;
    A(1, 0) = 4;
    A(1, 1) = 5;
    A(1, 2) = 6;

    vec x{1.0, 1.0, 1.0};
    vec y(2);

    matvec(A, x, y);
    EXPECT_DOUBLE_EQ(y[0], 6.0);
    EXPECT_DOUBLE_EQ(y[1], 15.0);
}

TEST(mat, Matmul) {
    mat A(2, 3);
    A(0, 0) = 1;
    A(0, 1) = 2;
    A(0, 2) = 3;
    A(1, 0) = 4;
    A(1, 1) = 5;
    A(1, 2) = 6;

    mat B(3, 2);
    B(0, 0) = 1;
    B(0, 1) = 2;
    B(1, 0) = 3;
    B(1, 1) = 4;
    B(2, 0) = 5;
    B(2, 1) = 6;

    mat C(2, 2);
    matmul(A, B, C);

    EXPECT_DOUBLE_EQ(C(0, 0), 22.0);
    EXPECT_DOUBLE_EQ(C(0, 1), 28.0);
    EXPECT_DOUBLE_EQ(C(1, 0), 49.0);
    EXPECT_DOUBLE_EQ(C(1, 1), 64.0);
}

TEST(mat, Matadd) {
    mat A(2, 2, 1.0);
    mat B(2, 2, 2.0);
    mat C(2, 2);

    matadd(2.0, A, 3.0, B, C);
    EXPECT_DOUBLE_EQ(C(0, 0), 8.0);
}

TEST(mat, RowAndElementScaling) {
    mat A(2, 2, 1.0);
    const std::vector<real> weights{2.0, 4.0};
    scale_rows(A, weights);
    EXPECT_DOUBLE_EQ(A(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(A(1, 0), 4.0);
    divide_rows(A, weights);
    EXPECT_DOUBLE_EQ(A(0, 1), 1.0);
    EXPECT_DOUBLE_EQ(A(1, 0), 1.0);

    vec x{3.0, 5.0};
    scale_elements(x, weights);
    EXPECT_DOUBLE_EQ(x[0], 6.0);
    EXPECT_DOUBLE_EQ(x[1], 20.0);
    divide_elements(x, weights);
    EXPECT_DOUBLE_EQ(x[0], 3.0);
    EXPECT_DOUBLE_EQ(x[1], 5.0);
}

TEST(mat, RelativeSymmetryError) {
    mat A(2, 2, 0.0);
    A(0, 1) = 100.0;
    A(1, 0) = 100.01;
    EXPECT_NEAR(linear::relative_symmetry_error(A), 0.01 / 100.01, 1e-14);
}

// Backend correctness: every backend must produce the same result as
// backend::seq

static mat make_test_matrix(idx rows, idx cols) {
    mat A(rows, cols);
    for (idx i = 0; i < rows; ++i) {
        for (idx j = 0; j < cols; ++j) {
            A(i, j) = static_cast<real>((i * cols) + j + 1);
        }
    }
    return A;
}

static vec make_test_vector(idx n) {
    vec v(n);
    for (idx i = 0; i < n; ++i) {
        v[i] = static_cast<real>(i + 1);
    }
    return v;
}

// `kernel::gemm` walks its register tile over m and n and drops to scalar code
// for whatever is left over, so a shape that divides the tile evenly exercises
// none of that. These cases deliberately land on both remainders, and on
// m != n != k -- the last is what caught a stride bug in the hand-written SIMD
// product this kernel replaced, which read A with the wrong leading dimension
// and so was correct only on square inputs.
TEST(Gemm, MatchesNaiveProductOnTileRemainders) {
    struct Shape {
        idx m, n, k;
    };
    for (const Shape s : {Shape{32, 32, 32}, Shape{17, 23, 11}, Shape{4, 64, 4},
                          Shape{64, 4, 64}, Shape{1, 1, 1}, Shape{5, 3, 7},
                          Shape{33, 65, 17}}) {
        const mat A = make_test_matrix(s.m, s.k);
        const mat B = make_test_matrix(s.k, s.n);
        mat C(s.m, s.n, 0.0);
        seq::matmul(A, B, C);

        for (idx i = 0; i < s.m; ++i) {
            for (idx j = 0; j < s.n; ++j) {
                real expected = 0.0;
                for (idx p = 0; p < s.k; ++p) {
                    expected += A(i, p) * B(p, j);
                }
                // The tiled kernel sums in the same order as this loop, so it
                // should agree to the last bit, not merely to a tolerance.
                EXPECT_DOUBLE_EQ(C(i, j), expected)
                    << "at (" << i << "," << j << ") for " << s.m << "x" << s.n << "x" << s.k;
            }
        }
    }
}

// beta != 0 accumulates into C, which the tiled path reaches by a separate
// prologue from the beta == 0 overwrite.
TEST(Gemm, ScalesAndAccumulatesIntoC) {
    const idx n = 20;
    const mat A = make_test_matrix(n, n);
    const mat B = make_test_matrix(n, n);
    mat product(n, n, 0.0);
    seq::matmul(A, B, product);

    mat C(n, n, 3.0);
    kernel::gemm(C.data(), A.data(), B.data(), real(2), real(-1), n, n, n);
    for (idx i = 0; i < n; ++i) {
        for (idx j = 0; j < n; ++j) {
            EXPECT_DOUBLE_EQ(C(i, j), (2.0 * product(i, j)) - 3.0);
        }
    }
}


TEST(MatrixExpr, ValueReturningMatchesOutParameterForm) {
    mat A(2, 3, 0.0);
    mat B(3, 2, 0.0);
    for (idx i = 0; i < 2; ++i) {
        for (idx j = 0; j < 3; ++j) {
            A(i, j) = 1.0 + double(i) + 2.0 * double(j);
            B(j, i) = 0.5 - double(i) + double(j);
        }
    }
    vec x{1.0, -2.0, 3.0};

    mat expected_product(2, 2, 0.0);
    matmul(A, B, expected_product);
    const mat product = matmul(A, B);
    ASSERT_EQ(product.rows(), 2);
    ASSERT_EQ(product.cols(), 2);
    for (idx i = 0; i < 2; ++i) {
        for (idx j = 0; j < 2; ++j) {
            EXPECT_DOUBLE_EQ(product(i, j), expected_product(i, j));
        }
    }

    vec expected_image(2, 0.0);
    matvec(A, x, expected_image);
    const vec image = matvec(A, x);
    ASSERT_EQ(image.size(), 2);
    EXPECT_DOUBLE_EQ(image[0], expected_image[0]);
    EXPECT_DOUBLE_EQ(image[1], expected_image[1]);

    const mat sum = add(A, A);
    const mat difference = sub(A, A);
    for (idx i = 0; i < 2; ++i) {
        for (idx j = 0; j < 3; ++j) {
            EXPECT_DOUBLE_EQ(sum(i, j), 2.0 * A(i, j));
            EXPECT_DOUBLE_EQ(difference(i, j), 0.0);
        }
    }

    const vec vector_sum = add(x, x);
    const vec vector_difference = sub(x, x);
    for (idx i = 0; i < x.size(); ++i) {
        EXPECT_DOUBLE_EQ(vector_sum[i], 2.0 * x[i]);
        EXPECT_DOUBLE_EQ(vector_difference[i], 0.0);
    }
}

TEST(MatrixExpr, OperatorsAgreeWithNamedForms) {
    using namespace num::ops;

    mat A(2, 2, 0.0);
    A(0, 0) = 4.0;
    A(0, 1) = 1.0;
    A(1, 0) = 2.0;
    A(1, 1) = 3.0;
    const vec x{1.0, 2.0};

    const mat product = A * A;
    const mat named_product = matmul(A, A);
    const vec image = A * x;
    const vec named_image = matvec(A, x);
    const mat sum = A + A;
    const mat difference = A - A;
    const vec vector_sum = x + x;
    const vec vector_difference = x - x;

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
    EXPECT_THROW((void)matmul(mat(2, 3, 1.0), mat(2, 3, 1.0)), std::invalid_argument);
    EXPECT_THROW((void)matvec(mat(2, 3, 1.0), vec(2, 1.0)), std::invalid_argument);
    EXPECT_THROW((void)add(mat(2, 2, 1.0), mat(3, 3, 1.0)), std::invalid_argument);
    EXPECT_THROW((void)sub(mat(2, 2, 1.0), mat(3, 3, 1.0)), std::invalid_argument);
    EXPECT_THROW((void)add(vec(2, 1.0), vec(3, 1.0)), std::invalid_argument);
    EXPECT_THROW((void)sub(vec(2, 1.0), vec(3, 1.0)), std::invalid_argument);
}
