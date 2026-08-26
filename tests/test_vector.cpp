#include "container/vector.hpp"
#include "container/vector_ops.hpp"
#include "kernel/raw.hpp"
#include <gtest/gtest.h>

using namespace num;

TEST(Vector, Construction) {
    Vector v(10);
    EXPECT_EQ(v.size(), 10);

    Vector v2(5, 3.0);
    for (idx i = 0; i < 5; ++i) {
        EXPECT_DOUBLE_EQ(v2[i], 3.0);
    }

    Vector v3{1.0, 2.0, 3.0};
    EXPECT_EQ(v3.size(), 3);
    EXPECT_DOUBLE_EQ(v3[1], 2.0);
}

TEST(Vector, CopyMove) {
    Vector v{1.0, 2.0, 3.0};
    Vector copy = v;
    EXPECT_DOUBLE_EQ(copy[0], 1.0);

    Vector moved = std::move(copy);
    EXPECT_EQ(moved.size(), 3);
}

TEST(Vector, Scale) {
    Vector v{1.0, 2.0, 3.0};
    scale(v, 2.0);
    EXPECT_DOUBLE_EQ(v[0], 2.0);
    EXPECT_DOUBLE_EQ(v[1], 4.0);
    EXPECT_DOUBLE_EQ(v[2], 6.0);
}

TEST(Vector, Add) {
    Vector x{1.0, 2.0, 3.0};
    Vector y{4.0, 5.0, 6.0};
    Vector z(3);
    add(x, y, z);
    EXPECT_DOUBLE_EQ(z[0], 5.0);
    EXPECT_DOUBLE_EQ(z[1], 7.0);
    EXPECT_DOUBLE_EQ(z[2], 9.0);
}

TEST(Vector, Axpy) {
    Vector x{1.0, 2.0, 3.0};
    Vector y{1.0, 1.0, 1.0};
    axpy(2.0, x, y);
    EXPECT_DOUBLE_EQ(y[0], 3.0);
    EXPECT_DOUBLE_EQ(y[1], 5.0);
    EXPECT_DOUBLE_EQ(y[2], 7.0);
}

TEST(Vector, Dot) {
    Vector x{1.0, 2.0, 3.0};
    Vector y{4.0, 5.0, 6.0};
    EXPECT_DOUBLE_EQ(dot(x, y), 32.0);

    const std::vector<real> sx{1.0, 2.0, 3.0};
    const std::vector<real> sy{4.0, 5.0, 6.0};
    EXPECT_DOUBLE_EQ(dot(std::span<const real>(sx), std::span<const real>(sy)), 32.0);
    EXPECT_THROW((void)dot(std::span<const real>(sx), std::span<const real>(sy).first(2)),
                 std::invalid_argument);
}

TEST(Vector, Norm) {
    Vector v{3.0, 4.0};
    EXPECT_DOUBLE_EQ(norm(v), 5.0);
}

TEST(RawKernel, GivensRotation) {
    double a = 3.0, b = 4.0;
    double c = 0.0, s = 0.0;
    kernel::raw::rotg(a, b, c, s);
    EXPECT_NEAR(c, 0.6, 1e-14);
    EXPECT_NEAR(s, 0.8, 1e-14);

    double x[2] = {3.0, 1.0};
    double y[2] = {4.0, 2.0};
    kernel::raw::rot(x, y, c, s, 2);
    EXPECT_NEAR(x[0], 5.0, 1e-14);
    EXPECT_NEAR(y[0], 0.0, 1e-14);
}

TEST(RawKernel, ElementwiseAndHadamard) {
    double x[3] = {2.0, 4.0, 8.0};
    double y[3] = {3.0, 2.0, 0.5};
    double z[3] = {0.0, 0.0, 0.0};

    kernel::raw::hadamard_mul(z, x, y, 3);
    EXPECT_DOUBLE_EQ(z[0], 6.0);
    EXPECT_DOUBLE_EQ(z[1], 8.0);
    EXPECT_DOUBLE_EQ(z[2], 4.0);

    kernel::raw::hadamard_div(z, x, y, 3);
    EXPECT_NEAR(z[0], 2.0 / 3.0, 1e-14);
    EXPECT_DOUBLE_EQ(z[1], 2.0);
    EXPECT_DOUBLE_EQ(z[2], 16.0);

    kernel::raw::inv(z, x, 3);
    EXPECT_DOUBLE_EQ(z[0], 0.5);
    EXPECT_DOUBLE_EQ(z[1], 0.25);
    EXPECT_DOUBLE_EQ(z[2], 0.125);

    kernel::raw::clamp(x, 3.0, 5.0, 3);
    EXPECT_DOUBLE_EQ(x[0], 3.0);
    EXPECT_DOUBLE_EQ(x[1], 4.0);
    EXPECT_DOUBLE_EQ(x[2], 5.0);
}

TEST(RawKernel, SpMVAndTranspose) {
    // 2x2 CSR matrix: [[1, 2], [3, 4]]
    double val[4] = {1.0, 2.0, 3.0, 4.0};
    idx row_ptr[3] = {0, 2, 4};
    idx col_idx[4] = {0, 1, 0, 1};
    double x[2] = {2.0, 3.0};
    double y[2] = {0.0, 0.0};

    kernel::raw::spmv(y, val, row_ptr, col_idx, x, 2);
    EXPECT_DOUBLE_EQ(y[0], 8.0);  // 1*2 + 2*3 = 8
    EXPECT_DOUBLE_EQ(y[1], 18.0); // 3*2 + 4*3 = 18

    // Fused spmv_axpy: y = 2.0 * A * x + 1.0 * y
    kernel::raw::spmv_axpy(y, 2.0, val, row_ptr, col_idx, x, 1.0, 2);
    EXPECT_DOUBLE_EQ(y[0], 24.0); // 2*8 + 8 = 24
    EXPECT_DOUBLE_EQ(y[1], 54.0); // 2*18 + 18 = 54

    // Dense transpose
    double A[6] = {1, 2, 3, 4, 5, 6}; // 2x3
    double B[6] = {0};                // 3x2
    kernel::raw::transpose(B, A, 2, 3);
    EXPECT_DOUBLE_EQ(B[0], 1.0);
    EXPECT_DOUBLE_EQ(B[1], 4.0);
    EXPECT_DOUBLE_EQ(B[2], 2.0);
    EXPECT_DOUBLE_EQ(B[3], 5.0);
    EXPECT_DOUBLE_EQ(B[4], 3.0);
    EXPECT_DOUBLE_EQ(B[5], 6.0);

    // Matvec transpose: y = A^T * x (where A is 2x3, x is 2x1 -> y is 3x1)
    double x2[2] = {1.0, 2.0};
    double y3[3] = {0};
    kernel::raw::matvec_transpose(y3, A, x2, 2, 3);
    EXPECT_DOUBLE_EQ(y3[0], 9.0);  // 1*1 + 4*2 = 9
    EXPECT_DOUBLE_EQ(y3[1], 12.0); // 2*1 + 5*2 = 12
    EXPECT_DOUBLE_EQ(y3[2], 15.0); // 3*1 + 6*2 = 15
}
