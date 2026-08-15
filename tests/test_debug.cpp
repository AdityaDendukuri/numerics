#include "numerics.hpp"
#include <gtest/gtest.h>
#include <cmath>

TEST(DebugCheck, DimensionMismatch) {
  EXPECT_THROW(
      num::debug::check_dim(5, 3, "test_vector"),
      std::invalid_argument);
}

TEST(DebugCheck, NonFiniteValueError) {
  double data[3] = {1.0, NAN, 3.0};
  EXPECT_THROW(
      num::debug::check_finite(data, 3, "test_array"),
      std::invalid_argument);
}

TEST(DebugCheck, FalseSPDAssertionCaughtAtRuntime) {
  // Create a 2x2 matrix with negative diagonal entry (indefinite/negative definite)
  num::Matrix A(2, 2, 0.0);
  A(0, 0) = -5.0;
  A(1, 1) = 1.0;

  num::operators::DenseOp Aop(A);


  // assume_spd() throws a PropertyError because sampled x^T A x is <= 0!
  EXPECT_THROW(
      num::operators::assume_spd(Aop),
      std::invalid_argument);
}
