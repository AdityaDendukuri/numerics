#include "container/matrix.hpp"
#include "container/matrix_expr.hpp"
#include "container/matrix_ops.hpp"
#include "kernel/factor.hpp"
#include "linear/matrix_properties.hpp"
#include "linear/matrix_utils.hpp"
#include "omp/matrix_ops.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <vector>

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
    for (const Shape s : {Shape{32, 32, 32}, Shape{17, 23, 11}, Shape{4, 64, 4}, Shape{64, 4, 64},
                          Shape{1, 1, 1}, Shape{5, 3, 7}, Shape{33, 65, 17}}) {
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

namespace {
// Reference product over strided operands, summed in p order.
void naive_gemm(real *C, idx ldc, const real *A, idx a_rows, idx a_cols, const real *B, idx b_rows,
                idx b_cols, real alpha, real beta, idx m, idx n, idx k) {
    for (idx i = 0; i < m; ++i) {
        for (idx j = 0; j < n; ++j) {
            real sum = 0.0;
            for (idx p = 0; p < k; ++p) {
                sum += A[(i * a_rows) + (p * a_cols)] * B[(p * b_rows) + (j * b_cols)];
            }
            C[(i * ldc) + j] = (alpha * sum) + (beta * C[(i * ldc) + j]);
        }
    }
}
} // namespace

// Leading dimensions larger than the logical extent: every operand is a window
// into a bigger buffer, so a packer that assumed `lda == k` would read the
// wrong column.
TEST(Gemm, HonoursLeadingDimensions) {
    const idx m = 13, n = 11, k = 9, lda = 20, ldb = 17, ldc = 15;
    const mat A = make_test_matrix(m, lda);
    const mat B = make_test_matrix(k, ldb);
    mat C = make_test_matrix(m, ldc);
    mat expected = C;
    naive_gemm(expected.data(), ldc, A.data(), lda, 1, B.data(), ldb, 1, 1.5, -0.5, m, n, k);
    kernel::gemm(C.data(), ldc, A.data(), lda, B.data(), ldb, real(1.5), real(-0.5), m, n, k);
    for (idx i = 0; i < m; ++i) {
        for (idx j = 0; j < ldc; ++j) {
            EXPECT_NEAR(C(i, j), expected(i, j), 1e-12) << "at (" << i << "," << j << ")";
        }
    }
}

// An inner dimension longer than one packed panel exercises the outer k loop,
// where partial sums from successive panels are added into C.
TEST(Gemm, AccumulatesAcrossPanels) {
    using cfg = kernel::gemm_config<real>;
    const idx m = 7, n = 5, k = (2 * cfg::kc) + 3;
    const mat A = make_test_matrix(m, k);
    const mat B = make_test_matrix(k, n);
    mat C(m, n, 0.0);
    mat expected(m, n, 0.0);
    naive_gemm(expected.data(), n, A.data(), k, 1, B.data(), n, 1, 1.0, 0.0, m, n, k);
    kernel::gemm(C.data(), A.data(), B.data(), real(1), real(0), m, n, k);
    for (idx i = 0; i < m; ++i) {
        for (idx j = 0; j < n; ++j) {
            EXPECT_NEAR(C(i, j), expected(i, j), 1e-9 * std::abs(expected(i, j)));
        }
    }
}

// Caller-provided workspace is the allocation-free path; it must produce the
// same result as the static-buffer overload, and `gemm_workspace` must never
// exceed the configured maximum.
TEST(Gemm, ExplicitWorkspaceMatchesStaticPath) {
    using cfg = kernel::gemm_config<real>;
    const idx m = 30, n = 14, k = 22;
    EXPECT_LE(kernel::gemm_workspace<real>(m, n, k), cfg::workspace);
    EXPECT_LE(kernel::gemm_workspace<real>(5000, 5000, 5000), cfg::workspace);
    std::vector<real> work(kernel::gemm_workspace<real>(m, n, k));
    const mat A = make_test_matrix(m, k);
    const mat B = make_test_matrix(k, n);
    mat with_work(m, n, 0.0), with_static(m, n, 0.0);
    kernel::gemm(with_work.data(), n, A.data(), k, B.data(), n, real(1), real(0), m, n, k,
                 work.data());
    kernel::gemm(with_static.data(), n, A.data(), k, B.data(), n, real(1), real(0), m, n, k);
    for (idx i = 0; i < m; ++i) {
        for (idx j = 0; j < n; ++j) {
            EXPECT_DOUBLE_EQ(with_work(i, j), with_static(i, j));
        }
    }
}

// The transposed-left product shares the packed core with `gemm`; the strides
// are swapped rather than the data copied.
TEST(Gemm, TransposeLeftMatchesNaive) {
    const idx rows = 19, a_cols = 10, b_cols = 13;
    const mat A = make_test_matrix(rows, a_cols);
    const mat B = make_test_matrix(rows, b_cols);
    mat C(a_cols, b_cols, 1.0);
    mat expected = C;
    naive_gemm(expected.data(), b_cols, A.data(), 1, a_cols, B.data(), b_cols, 1, 0.5, 2.0, a_cols,
               b_cols, rows);
    kernel::gemm_transpose_left(C.data(), b_cols, A.data(), a_cols, B.data(), b_cols, real(0.5),
                                real(2), rows, a_cols, b_cols);
    for (idx i = 0; i < a_cols; ++i) {
        for (idx j = 0; j < b_cols; ++j) {
            EXPECT_NEAR(C(i, j), expected(i, j), 1e-12);
        }
    }
}

// `syrk_lower` may write only on and below the diagonal: blocked Cholesky
// factors in place and keeps live data in the strict upper triangle.
TEST(Gemm, SyrkLowerTouchesOnlyLowerTriangle) {
    using cfg = kernel::gemm_config<real>;
    const idx rows = (3 * cfg::nr) + 5, columns = 9; // several strips plus a remainder
    const mat A = make_test_matrix(rows, columns);
    mat C(rows, rows, 7.0);
    kernel::syrk_lower(C.data(), rows, A.data(), columns, real(-1), real(0.5), rows, columns);
    for (idx i = 0; i < rows; ++i) {
        for (idx j = 0; j < rows; ++j) {
            if (j > i) {
                EXPECT_DOUBLE_EQ(C(i, j), 7.0)
                    << "upper entry written at (" << i << "," << j << ")";
                continue;
            }
            real sum = 0.0;
            for (idx p = 0; p < columns; ++p) {
                sum += A(i, p) * A(j, p);
            }
            EXPECT_NEAR(C(i, j), (-1.0 * sum) + 3.5, 1e-12) << "at (" << i << "," << j << ")";
        }
    }
}

// The OpenMP product shares packed panels across threads; it must agree with
// the sequential kernel on shapes that overhang every blocking level.
TEST(Gemm, OmpMatmulMatchesSequential) {
    using cfg = kernel::gemm_config<real>;
    const idx m = cfg::mc + 5, n = cfg::nr + 3, k = cfg::kc + 7;
    const mat A = make_test_matrix(m, k);
    const mat B = make_test_matrix(k, n);
    mat sequential(m, n, 0.0), parallel(m, n, 0.0);
    seq::matmul(A, B, sequential);
    omp::matmul(A, B, parallel);
    for (idx i = 0; i < m; ++i) {
        for (idx j = 0; j < n; ++j) {
            EXPECT_DOUBLE_EQ(parallel(i, j), sequential(i, j));
        }
    }
}

// The four triangular solves are blocked over the packed gemm: each diagonal
// block by substitution, the rest by a rank-`trsm_block` update. Shapes here
// straddle the block boundary and every right-hand-side count the row batch
// in the right-side solve can leave over.
TEST(Trsm, AllVariantsInvertTheirProducts) {
    for (const idx n : {idx{1}, idx{5}, idx{63}, idx{64}, idx{65}, idx{200}}) {
        for (const idx nrhs : {idx{1}, idx{3}, idx{17}, idx{40}}) {
            mat L(n, n, 0.0);
            for (idx i = 0; i < n; ++i) {
                for (idx j = 0; j <= i; ++j) {
                    L(i, j) = i == j ? 2.0 + static_cast<real>(i % 3)
                                     : 0.01 * std::sin(1.0 + static_cast<real>(i * j));
                }
            }
            const mat X = make_test_matrix(n, nrhs);
            const auto expect_equal = [&](const mat &Y, const mat &reference, const char *what) {
                for (idx i = 0; i < Y.rows(); ++i) {
                    for (idx j = 0; j < Y.cols(); ++j) {
                        EXPECT_NEAR(Y(i, j), reference(i, j), 1e-9)
                            << what << " n=" << n << " nrhs=" << nrhs << " at (" << i << "," << j
                            << ")";
                    }
                }
            };

            mat B(n, nrhs, 0.0); // L X
            for (idx i = 0; i < n; ++i)
                for (idx r = 0; r < nrhs; ++r)
                    for (idx j = 0; j <= i; ++j)
                        B(i, r) += L(i, j) * X(j, r);
            mat Y = B;
            kernel::trsm_lower_inplace(Y.data(), nrhs, L.data(), n, nrhs);
            expect_equal(Y, X, "trsm_lower_inplace");

            B = mat(n, nrhs, 0.0); // L^T X
            for (idx i = 0; i < n; ++i)
                for (idx r = 0; r < nrhs; ++r)
                    for (idx k = i; k < n; ++k)
                        B(i, r) += L(k, i) * X(k, r);
            Y = B;
            kernel::trsm_lower_transpose_inplace(Y.data(), nrhs, L.data(), n, nrhs);
            expect_equal(Y, X, "trsm_lower_transpose_inplace");

            B = X; // (unit L) X
            for (idx i = 0; i < n; ++i)
                for (idx r = 0; r < nrhs; ++r)
                    for (idx j = 0; j < i; ++j)
                        B(i, r) += L(i, j) * X(j, r);
            Y = B;
            kernel::trsm_unit_lower_inplace(Y.data(), nrhs, L.data(), n, n, nrhs);
            expect_equal(Y, X, "trsm_unit_lower_inplace");

            const mat Xr = make_test_matrix(nrhs, n); // Xr L^T, rows independent
            mat Br(nrhs, n, 0.0);
            for (idx r = 0; r < nrhs; ++r)
                for (idx j = 0; j < n; ++j)
                    for (idx k = 0; k <= j; ++k)
                        Br(r, j) += Xr(r, k) * L(j, k);
            Y = Br;
            kernel::trsm_lower_transpose_right_inplace(Y.data(), n, L.data(), n, nrhs, n);
            expect_equal(Y, Xr, "trsm_lower_transpose_right_inplace");

            // The upper forms are the lower ones with the strides swapped: U = L^T.
            mat U(n, n, 0.0);
            for (idx i = 0; i < n; ++i)
                for (idx j = 0; j < n; ++j)
                    U(i, j) = L(j, i);
            B = mat(n, nrhs, 0.0); // U X
            for (idx i = 0; i < n; ++i)
                for (idx r = 0; r < nrhs; ++r)
                    for (idx k = i; k < n; ++k)
                        B(i, r) += U(i, k) * X(k, r);
            Y = B;
            kernel::trsm_upper_inplace(Y.data(), nrhs, U.data(), n, n, nrhs);
            expect_equal(Y, X, "trsm_upper_inplace");

            B = mat(n, nrhs, 0.0); // U^T X
            for (idx i = 0; i < n; ++i)
                for (idx r = 0; r < nrhs; ++r)
                    for (idx k = 0; k <= i; ++k)
                        B(i, r) += U(k, i) * X(k, r);
            Y = B;
            kernel::trsm_upper_transpose_inplace(Y.data(), nrhs, U.data(), n, n, nrhs);
            expect_equal(Y, X, "trsm_upper_transpose_inplace");

            B = X; // (unit L)^T X
            for (idx i = 0; i < n; ++i)
                for (idx r = 0; r < nrhs; ++r)
                    for (idx k = i + 1; k < n; ++k)
                        B(i, r) += L(k, i) * X(k, r);
            Y = B;
            kernel::trsm_unit_lower_transpose_inplace(Y.data(), nrhs, L.data(), n, n, nrhs);
            expect_equal(Y, X, "trsm_unit_lower_transpose_inplace");
        }
    }
}

// Blocked Cholesky recurses on diagonal blocks wider than the trsm block;
// sizes here cross both the panel and the recursion boundary.
TEST(Trsm, BlockedCholeskyReconstructsAndZeroesUpperTriangle) {
    for (const idx n : {idx{1}, idx{7}, idx{64}, idx{65}, idx{257}, idx{600}}) {
        mat A(n, n, 0.0);
        for (idx i = 0; i < n; ++i)
            for (idx j = 0; j < n; ++j)
                A(i, j) =
                    i == j ? static_cast<real>(n) + 1.0 : 1.0 / (1.0 + static_cast<real>(i + j));
        mat L = A;
        ASSERT_TRUE(kernel::cholesky_blocked(L.data(), n)) << "n=" << n;
        for (idx i = 0; i < n; ++i) {
            for (idx j = 0; j < n; ++j) {
                if (j > i) {
                    EXPECT_EQ(L(i, j), 0.0) << "n=" << n;
                    continue;
                }
                real sum = 0.0;
                for (idx k = 0; k <= j; ++k)
                    sum += L(i, k) * L(j, k);
                EXPECT_NEAR(sum, A(i, j), 1e-9 * static_cast<real>(n)) << "n=" << n;
            }
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
