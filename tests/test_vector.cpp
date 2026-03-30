#include <gtest/gtest.h>
#include "core/vector.hpp"

using namespace num;

TEST(Vector, Construction) {
    Vector v(10);
    EXPECT_EQ(v.size(), 10);

    Vector v2(5, 3.0);
    for (idx i = 0; i < 5; ++i) EXPECT_DOUBLE_EQ(v2[i], 3.0);

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
}

TEST(Vector, Norm) {
    Vector v{3.0, 4.0};
    EXPECT_DOUBLE_EQ(norm(v), 5.0);
}
