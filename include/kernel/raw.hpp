/// @file kernel/raw.hpp
/// @brief Tier-1 kernel: raw-pointer, inline, zero-overhead inner loops.
///
/// Raw kernels assume non-owning, caller-sized buffers and do not allocate.
#pragma once

#include "core/types.hpp"
#include <cmath>
#include <concepts>

#if defined(__GNUC__) || defined(__clang__)
  #define NUM_K_AINLINE  [[gnu::always_inline]] inline
  #define NUM_K_RESTRICT __restrict__
  #define NUM_K_IVDEP    _Pragma("GCC ivdep")
#else
  #define NUM_K_AINLINE inline
  #define NUM_K_RESTRICT
  #define NUM_K_IVDEP
#endif

namespace num::kernel::raw {

/// @brief x[i] *= alpha
template<std::floating_point T>
NUM_K_AINLINE void scale(T* NUM_K_RESTRICT x, T alpha, idx n) noexcept {
  NUM_K_IVDEP
  for (idx i = 0; i < n; ++i) {
    x[i] *= alpha;
  }
}

/// @brief y[i] += alpha * x[i]
template<std::floating_point T>
NUM_K_AINLINE void axpy(T* NUM_K_RESTRICT y,
                        const T* NUM_K_RESTRICT x,
                        T alpha,
                        idx n) noexcept {
  NUM_K_IVDEP
  for (idx i = 0; i < n; ++i) {
    y[i] += alpha * x[i];
  }
}

/// @brief y[i] = a*x[i] + b*y[i].
template<std::floating_point T>
NUM_K_AINLINE void axpby(T* NUM_K_RESTRICT y,
                         const T* NUM_K_RESTRICT x,
                         T a,
                         T b,
                         idx n) noexcept {
  NUM_K_IVDEP
  for (idx i = 0; i < n; ++i) {
    y[i] = (a * x[i]) + (b * y[i]);
  }
}

/// @brief z[i] = a*x[i] + b*y[i].
template<std::floating_point T>
NUM_K_AINLINE void axpbyz(T* NUM_K_RESTRICT z,
                          const T* NUM_K_RESTRICT x,
                          const T* NUM_K_RESTRICT y,
                          T a,
                          T b,
                          idx n) noexcept {
  NUM_K_IVDEP
  for (idx i = 0; i < n; ++i) {
    z[i] = (a * x[i]) + (b * y[i]);
  }
}

/// @brief dot product: return sum x[i] * y[i]
template<std::floating_point T>
[[nodiscard]] NUM_K_AINLINE T dot(const T* NUM_K_RESTRICT x,
                                  const T* NUM_K_RESTRICT y,
                                  idx n) noexcept {
  T s = T(0);
  NUM_K_IVDEP
  for (idx i = 0; i < n; ++i) {
    s += x[i] * y[i];
  }
  return s;
}

/// @brief sum x[i]^2  (no sqrt; use for convergence checks to avoid sqrt cost)
template<std::floating_point T>
[[nodiscard]] NUM_K_AINLINE T norm_sq(const T* NUM_K_RESTRICT x, idx n) noexcept {
  T s = T(0);
  NUM_K_IVDEP
  for (idx i = 0; i < n; ++i) {
    s += x[i] * x[i];
  }
  return s;
}

/// @brief Euclidean norm: sqrt(sum x[i]^2)
template<std::floating_point T>
[[nodiscard]] NUM_K_AINLINE T norm(const T* NUM_K_RESTRICT x, idx n) noexcept {
  return std::sqrt(norm_sq(x, n));
}

/// @brief L1 norm: sum |x[i]|
template<std::floating_point T>
[[nodiscard]] NUM_K_AINLINE T l1_norm(const T* NUM_K_RESTRICT x, idx n) noexcept {
  T s = T(0);
  NUM_K_IVDEP
  for (idx i = 0; i < n; ++i) {
    s += std::abs(x[i]);
  }
  return s;
}

/// @brief L-infinity norm: max |x[i]|
template<std::floating_point T>
[[nodiscard]] NUM_K_AINLINE T linf_norm(const T* NUM_K_RESTRICT x, idx n) noexcept {
  T mx = T(0);
  for (idx i = 0; i < n; ++i) {
    const T v = std::abs(x[i]);
    if (v > mx) {
      mx = v;
    }
  }
  return mx;
}

/// @brief Scalar sum: return sum x[i]
template<std::floating_point T>
[[nodiscard]] NUM_K_AINLINE T sum(const T* NUM_K_RESTRICT x, idx n) noexcept {
  T s = T(0);
  NUM_K_IVDEP
  for (idx i = 0; i < n; ++i) {
    s += x[i];
  }
  return s;
}

/// @brief y[i] = sum_j A[i*n + j] * x[j]  (m x n row-major matrix)
///
/// y must be pre-allocated with size m. y and x must not alias A.
template<std::floating_point T>
NUM_K_AINLINE void matvec(T* NUM_K_RESTRICT y,
                          const T* NUM_K_RESTRICT A,
                          const T* NUM_K_RESTRICT x,
                          idx m,
                          idx n) noexcept {
  for (idx i = 0; i < m; ++i) {
    T s = T(0);
    const T* row = A + (i * n);
    NUM_K_IVDEP
    for (idx j = 0; j < n; ++j) {
      s += row[j] * x[j];
    }
    y[i] = s;
  }
}

/// @brief Rank-1 update: A[i*n + j] += alpha * x[i] * y[j]  (m x n row-major)
///
/// Inner loop over j is independent and vectorizable.
template<std::floating_point T>
NUM_K_AINLINE void ger(T* NUM_K_RESTRICT A,
                       const T* NUM_K_RESTRICT x,
                       const T* NUM_K_RESTRICT y,
                       T alpha,
                       idx m,
                       idx n) noexcept {
  for (idx i = 0; i < m; ++i) {
    T* NUM_K_RESTRICT row = A + (i * n);
    const T axi = alpha * x[i];
    NUM_K_IVDEP
    for (idx j = 0; j < n; ++j) {
      row[j] += axi * y[j];
    }
  }
}

/// @brief Forward substitution: solve Lx = b, L lower triangular (n x n,
/// row-major).
///
/// x must be pre-allocated with size n. b and x must not alias each other.
template<std::floating_point T>
NUM_K_AINLINE void trsv_lower(T* NUM_K_RESTRICT x,
                              const T* NUM_K_RESTRICT L,
                              const T* NUM_K_RESTRICT b,
                              idx n) noexcept {
  for (idx i = 0; i < n; ++i) {
    T s = b[i];
    const T* row = L + (i * n);
    for (idx j = 0; j < i; ++j) {
      s -= row[j] * x[j];
    }
    x[i] = s / row[i];
  }
}

/// @brief Back substitution: solve Ux = b, U upper triangular (n x n,
/// row-major).
///
/// x must be pre-allocated with size n. b and x must not alias each other.
template<std::floating_point T>
NUM_K_AINLINE void trsv_upper(T* NUM_K_RESTRICT x,
                              const T* NUM_K_RESTRICT U,
                              const T* NUM_K_RESTRICT b,
                              idx n) noexcept {
  for (idx i = n; i-- > 0;) {
    T s = b[i];
    const T* row = U + (i * n);
    for (idx j = i + 1; j < n; ++j) {
      s -= row[j] * x[j];
    }
    x[i] = s / row[i];
  }
}

} // namespace num::kernel::raw
