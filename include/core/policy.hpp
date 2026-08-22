/// @file policy.hpp
/// @brief Backend enum and default backend selection.
#pragma once

#include <cstdint>

namespace num {

enum class Backend : std::uint8_t {
    seq,
    blocked,
    simd,
    blas,
    omp,
    gpu,
    lapack,
};

inline constexpr Backend seq = Backend::seq;
inline constexpr Backend blocked = Backend::blocked;
inline constexpr Backend simd = Backend::simd;
inline constexpr Backend blas = Backend::blas;
inline constexpr Backend omp = Backend::omp;
inline constexpr Backend gpu = Backend::gpu;
inline constexpr Backend lapack = Backend::lapack;

inline constexpr bool has_blas =
#if defined(NUMERICS_HAS_BLAS)
    true;
#else
    false;
#endif

inline constexpr bool has_lapack =
#if defined(NUMERICS_HAS_LAPACK)
    true;
#else
    false;
#endif

inline constexpr bool has_omp =
#if defined(NUMERICS_HAS_OMP)
    true;
#else
    false;
#endif

inline constexpr bool has_simd =
#if defined(NUMERICS_HAS_SIMD)
    true;
#else
    false;
#endif

inline constexpr Backend default_backend =
#if defined(NUMERICS_HAS_BLAS)
    Backend::blas;
#elif defined(NUMERICS_HAS_OMP)
    Backend::omp;
#elif defined(NUMERICS_HAS_SIMD)
                    Backend::simd;
#else
                    Backend::blocked;
#endif

inline constexpr Backend best_backend = default_backend;

inline constexpr Backend lapack_backend =
#if defined(NUMERICS_HAS_LAPACK)
    Backend::lapack;
#elif defined(NUMERICS_HAS_OMP)
    Backend::omp;
#else
    Backend::seq;
#endif

} // namespace num
