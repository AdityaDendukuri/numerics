/// @file types.hpp
/// @brief Core type definitions
#pragma once

#include <complex>
#include <cstddef>

namespace num {

using real = double;
using idx  = std::size_t;
using cplx = std::complex<real>;

/// @brief Cast any integer to idx without a verbose static_cast.
template<class T>
constexpr idx to_idx(T x) noexcept { return static_cast<idx>(x); }

} // namespace num
