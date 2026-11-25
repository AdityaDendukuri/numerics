/// @file core/policy.hpp
/// @brief Build capabilities and the default backend.
#pragma once

#include "core/types.hpp"
#include <concepts>
#include <cstdint>
#include <type_traits>

namespace num {

// Build capabilities

/// @brief Whether the build links a BLAS.
inline constexpr bool has_blas =
#if defined(NUMERICS_HAS_BLAS)
    true;
#else
    false;
#endif

/// @brief Whether the build links LAPACKE.
inline constexpr bool has_lapack =
#if defined(NUMERICS_HAS_LAPACK)
    true;
#else
    false;
#endif

// The untagged dense factorizations take LAPACK only on an optimized BLAS.
// NUMERICS_LAPACK_REFERENCE marks reference LAPACK, which the kernel outperforms.
#if defined(NUMERICS_HAS_LAPACK) && !defined(NUMERICS_LAPACK_REFERENCE)
#define NUMERICS_LAPACK_DEFAULT 1
#endif

/// @brief Matrix order above which the untagged LU uses LAPACK.
inline constexpr idx lapack_factor_threshold = 768;

/// @brief Whether the untagged SVD, LU inverse and large LU use LAPACK.
inline constexpr bool lapack_default =
#if defined(NUMERICS_LAPACK_DEFAULT)
    true;
#else
    false;
#endif

/// @brief Whether the build uses OpenMP.
inline constexpr bool has_omp =
#if defined(NUMERICS_HAS_OMP)
    true;
#else
    false;
#endif

// whether the build enabled a wider instruction set than the baseline (`-mavx2 -mfma`).
// Only the FFT's intrinsic path reads it; `num::kernel` relies on the compiler.
/// @brief Whether the build enables SIMD beyond the target baseline.
inline constexpr bool has_simd =
#if defined(NUMERICS_HAS_SIMD)
    true;
#else
    false;
#endif

/// @brief Whether the build targets CUDA.
inline constexpr bool has_cuda =
#if defined(NUMERICS_HAS_CUDA)
    true;
#else
    false;
#endif

// The one compile-time default

// forward-declare every backend so `accel` can name whichever is available; each backend's
// headers reopen its namespace. `num::seq` is the fallback, since `num::kernel` knows only
// raw pointers.
namespace kernel {}
namespace seq {}
namespace omp {}
namespace blas {}
namespace lapack {}
namespace cuda {}

#if defined(NUMERICS_HAS_CUDA)
namespace accel = cuda;
#elif defined(NUMERICS_HAS_BLAS)
namespace accel = blas;
#elif defined(NUMERICS_HAS_OMP)
namespace accel = omp;
#else
namespace accel = seq;
#endif

} // namespace num
