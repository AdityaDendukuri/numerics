/// @file integer_pow.hpp
/// @brief Compile-time integer exponentiation via repeated squaring
#pragma once

namespace num {

/// @brief Compute x^N at compile time via repeated squaring.
template<int N, typename T>
constexpr T ipow(T x) noexcept {
  static_assert(N >= 0, "ipow: exponent must be non-negative");
  if constexpr (N == 0) {
    return T(1);
}
  if constexpr (N == 1) {
    return x;
}
  if constexpr (N % 2 == 0) {
    const T half = ipow<N / 2>(x);
    return half * half;
  } else {
    return x * ipow<N - 1>(x);
  }
}

} // namespace num
