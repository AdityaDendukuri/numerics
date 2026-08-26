/// @file types.hpp
/// @brief Core type definitions
#pragma once

#include <complex>
#include <cstddef>
#include <functional>

namespace num {

using real = double;
using idx = std::size_t;
using cplx = std::complex<real>;

/// @brief Cast any integer to idx without a verbose static_cast.
template <class T>
constexpr idx to_idx(T x) noexcept {
    return static_cast<idx>(x);
}

/// Scalar callback \f$f(x)\f$.
using ScalarFn = std::function<real(real)>;

/// Vector callback writing \f$f(t, y)\f$ into a caller-provided buffer.
using VectorFn = std::function<void(real, real *, real *)>;

} // namespace num
