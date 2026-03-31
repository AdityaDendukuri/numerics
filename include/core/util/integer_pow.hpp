/// @file integer_pow.hpp
/// @brief Compile-time integer exponentiation via repeated squaring
///
/// @par Algorithm (CS theory: exponentiation by squaring)
///
/// Naive loop:   x^N needs N-1 multiplications.
/// Squaring:     x^N needs ceil(log2(N)) squarings + popcount(N)-1
/// multiplications
///               = O(log N) total  -- for N=7: 4 mults vs 6 naive.
///
/// The recursion:
///   ipow<0>(x) = 1
///   ipow<1>(x) = x
///   ipow<N>(x) = ipow<N/2>(x)^2          if N even
///   ipow<N>(x) = x * ipow<N-1>(x)        if N odd
///
/// The compiler evaluates the N/2 branch once (stores in a temp) and
/// squares it, so ipow<6> compiles to exactly 3 multiplications:
///   t  = x*x      (x^2)
///   t2 = t*x      (x^3)
///   return t2*t2  (x^6)
/// and ipow<7> adds one more:
///   return x * ipow<6>(x)  (x^7 in 4 mults total)
///
/// @par Primary use: Tait EOS with gamma=7
///   float r_pow = num::ipow<7>(ratio);   // 4 mults, zero branching
#pragma once

namespace num {

/// @brief Compute x^N at compile time via repeated squaring.
/// @tparam N  Non-negative integer exponent (must be a compile-time constant).
/// @tparam T  Arithmetic type (float, double, int, ...).
template<int N, typename T>
constexpr T ipow(T x) noexcept {
    static_assert(N >= 0, "ipow: exponent must be non-negative");
    if constexpr (N == 0)
        return T(1);
    if constexpr (N == 1)
        return x;
    if constexpr (N % 2 == 0) {
        const T half = ipow<N / 2>(x);
        return half * half;
    } else {
        return x * ipow<N - 1>(x);
    }
}

} // namespace num
