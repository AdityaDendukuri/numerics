#include "numerics.hpp"
#include <gtest/gtest.h>

TEST(Umbrella, IncludeHeader) {
  num::Vector v(5, 1.0);
  EXPECT_EQ(v.size(), 5u);
}
