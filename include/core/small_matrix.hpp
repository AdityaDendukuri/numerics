/// @file small_matrix.hpp
/// @brief Constexpr fixed-size stack-allocated matrix and Givens rotation.
///
/// These types complement the heap-allocated Matrix / Vector for small inner
/// kernels where heap traffic is undesirable:
///
///   SmallVec<N>         -- constexpr dense vector backed by std::array<real,N>
///   SmallMatrix<M,N>    -- constexpr row-major matrix backed by
///   std::array<real,M*N> GivensRotation      -- constexpr (c,s) pair; apply()
///   / apply_t() are branchless
///
/// Because these are literal types every operation can be evaluated at
/// compile time (C++17: arithmetic only; C++20+: also sqrt/hypot via
/// constexpr `<cmath>`).  At runtime the compiler places them entirely in
/// registers for small M, N.
///
/// Typical uses:
///   - GMRES: Hessenberg column updates via GivensRotation::from + apply
///   - Jacobi SVD / eig: 2x2 eigensystem kernel as SmallMatrix<2,2>
///   - Register-tile accumulation: SmallMatrix<4,4> replaces 16 named doubles
///   - Unit tests: known-answer matmuls computed at compile time
#pragma once

#include "core/types.hpp"
#include <algorithm>
#include <array>
#include <cmath>

namespace num {

// SmallVec<N>

/// @brief Constexpr fixed-size dense vector (stack-allocated).
template <idx N> struct SmallVec {
  std::array<real, N> data{};

  constexpr real &operator[](idx i) noexcept { return data[i]; }
  constexpr const real &operator[](idx i) const noexcept { return data[i]; }
  static constexpr idx size() noexcept { return N; }

  constexpr SmallVec &operator+=(const SmallVec &o) noexcept {
    for (idx i = 0; i < N; ++i)
      data[i] += o.data[i];
    return *this;
  }
  constexpr SmallVec &operator-=(const SmallVec &o) noexcept {
    for (idx i = 0; i < N; ++i)
      data[i] -= o.data[i];
    return *this;
  }
  constexpr SmallVec &operator*=(real s) noexcept {
    for (idx i = 0; i < N; ++i)
      data[i] *= s;
    return *this;
  }

  constexpr real dot(const SmallVec &o) const noexcept {
    real s = 0;
    for (idx i = 0; i < N; ++i)
      s += data[i] * o.data[i];
    return s;
  }
  /// @brief Sum of squares (avoid sqrt to stay constexpr in C++17).
  constexpr real norm_sq() const noexcept { return dot(*this); }
};

template <idx N>
constexpr SmallVec<N> operator+(SmallVec<N> a, const SmallVec<N> &b) noexcept {
  return a += b;
}
template <idx N>
constexpr SmallVec<N> operator*(real s, SmallVec<N> v) noexcept {
  return v *= s;
}

// SmallMatrix<M, N>

/// @brief Constexpr fixed-size row-major matrix (stack-allocated).
template <idx M, idx N> struct SmallMatrix {
  std::array<real, M * N> data{};

  constexpr real &operator()(idx i, idx j) noexcept { return data[i * N + j]; }
  constexpr const real &operator()(idx i, idx j) const noexcept {
    return data[i * N + j];
  }

  static constexpr idx rows() noexcept { return M; }
  static constexpr idx cols() noexcept { return N; }

  constexpr void fill(real v) noexcept { data.fill(v); }

  static constexpr SmallMatrix zeros() noexcept { return SmallMatrix{}; }

  static constexpr SmallMatrix identity() noexcept {
    static_assert(M == N, "identity() requires a square matrix");
    SmallMatrix m{};
    for (idx i = 0; i < M; ++i)
      m(i, i) = real(1);
    return m;
  }

  constexpr SmallMatrix<N, M> transposed() const noexcept {
    SmallMatrix<N, M> t{};
    for (idx i = 0; i < M; ++i)
      for (idx j = 0; j < N; ++j)
        t(j, i) = (*this)(i, j);
    return t;
  }

  /// @brief Matrix multiplication: (M x N) * (N x K) -> (M x K).
  template <idx K>
  constexpr SmallMatrix<M, K>
  operator*(const SmallMatrix<N, K> &B) const noexcept {
    SmallMatrix<M, K> C{};
    for (idx i = 0; i < M; ++i)
      for (idx k = 0; k < N; ++k)
        for (idx j = 0; j < K; ++j)
          C(i, j) += (*this)(i, k) * B(k, j);
    return C;
  }

  /// @brief Matrix-vector product: (MxN) * (N) -> (M).
  constexpr SmallVec<M> operator*(const SmallVec<N> &x) const noexcept {
    SmallVec<M> y{};
    for (idx i = 0; i < M; ++i)
      for (idx j = 0; j < N; ++j)
        y[i] += (*this)(i, j) * x[j];
    return y;
  }

  constexpr SmallMatrix &operator+=(const SmallMatrix &o) noexcept {
    for (idx k = 0; k < M * N; ++k)
      data[k] += o.data[k];
    return *this;
  }
  constexpr SmallMatrix &operator*=(real s) noexcept {
    for (idx k = 0; k < M * N; ++k)
      data[k] *= s;
    return *this;
  }
};

// GivensRotation

/// @brief Givens plane rotation: G = [c, s; -s, c].
///
/// Constructed so that G * [a; b]^T = [r; 0] with r = hypot(a, b).
///
/// @code
///   auto g = GivensRotation::from(h_jj, h_jj1);  // zero H[j,j+1] in GMRES
///   g.apply(g_j, g_j1);                           // rotate RHS vector too
/// @endcode
struct GivensRotation {
  real c, s;

  /// Compute (c,s) such that G*[a;b] = [r;0], r = hypot(a,b).
  /// Stable for any (a,b) including b==0.
  static constexpr GivensRotation from(real a, real b) noexcept {
    if (b == real(0))
      return {real(1), real(0)};
    // Use the standard stable form.  std::hypot is constexpr in GCC/Clang
    // as a built-in extension even in C++17; becomes standard in C++23.
    real r = std::sqrt(a * a + b * b);
    return {a / r, b / r};
  }

  /// Apply G from the left: [x; y] <- G * [x; y].
  constexpr void apply(real &x, real &y) const noexcept {
    real tmp = c * x + s * y;
    y = -s * x + c * y;
    x = tmp;
  }

  /// Apply G^T (inverse rotation): [x; y] <- G^T * [x; y].
  constexpr void apply_t(real &x, real &y) const noexcept {
    real tmp = c * x - s * y;
    y = s * x + c * y;
    x = tmp;
  }

  /// Return the 2x2 rotation matrix.
  constexpr SmallMatrix<2, 2> as_matrix() const noexcept {
    return SmallMatrix<2, 2>{{c, s, -s, c}};
  }
};

} // namespace num
