/// @file policy.hpp
/// @brief Backend enum for linear algebra operations.
///
/// Each module defines its own backend enum for the choices relevant to it.
/// This enum covers linalg (vector, matrix, solvers, factorization, eigen,
/// svd). Other modules define their own -- e.g. spectral/ uses FFTBackend.
///
///   num::seq      -- naive serial C++              (always available)
///   num::blocked  -- cache-blocked, no intrinsics  (compiler auto-vectorizes)
///   num::simd     -- hand-written SIMD             (AVX2 on x86, NEON on
///   AArch64) num::blas     -- cblas / LAPACKE               (optional,
///   NUMERICS_HAS_BLAS) num::omp      -- OpenMP parallel (optional,
///   NUMERICS_HAS_OMP) num::gpu      -- CUDA (optional, NUMERICS_HAS_CUDA)
#pragma once

namespace num {

/// @brief Selects which backend handles a linalg operation.
enum class Backend {
    seq,     ///< Naive textbook loops  -- always available
    blocked, ///< Cache-blocked; compiler auto-vectorizes inner loops
    simd,    ///< Hand-written SIMD intrinsics (AVX2 or NEON)
    blas,    ///< cblas  -- OpenBLAS, MKL, Apple Accelerate  (Level-1/2/3)
    omp,     ///< OpenMP parallel blocked loops
    gpu,     ///< CUDA  -- custom kernels or cuBLAS
    lapack,  ///< LAPACKE  -- industry-standard factorizations, SVD, eigen
};

// Convenience constants  -- use these at call sites:
//   matmul(A, B, C, num::blas);
inline constexpr Backend seq     = Backend::seq;
inline constexpr Backend blocked = Backend::blocked;
inline constexpr Backend simd    = Backend::simd;
inline constexpr Backend blas    = Backend::blas;
inline constexpr Backend omp     = Backend::omp;
inline constexpr Backend gpu     = Backend::gpu;
inline constexpr Backend lapack  = Backend::lapack;

// Compile-time capability flags
/// True when a BLAS/cblas library was found at configure time.
inline constexpr bool has_blas =
#if defined(NUMERICS_HAS_BLAS)
    true;
#else
    false;
#endif

/// True when LAPACKE was found at configure time.
inline constexpr bool has_lapack =
#if defined(NUMERICS_HAS_LAPACK)
    true;
#else
    false;
#endif

/// True when OpenMP was found at configure time.
inline constexpr bool has_omp =
#if defined(NUMERICS_HAS_OMP)
    true;
#else
    false;
#endif

/// True when a SIMD ISA was detected (AVX2 on x86-64, NEON on AArch64).
inline constexpr bool has_simd =
#if defined(NUMERICS_HAS_SIMD)
    true;
#else
    false;
#endif

// Default and best backends
//
// Full hierarchy for dense vector/matrix operations:
//   blas > omp > simd > blocked > seq
//
// Rationale:
//   blas    -- hardware-tuned BLAS (OpenBLAS / MKL / Accelerate): fastest for
//              large n; uses BLAS-3 blocking and multi-threading internally.
//   omp     -- our parallel blocked loops: good when BLAS is absent and the
//              machine has multiple cores; thread overhead amortises for n >
//              ~256.
//   simd    -- our hand-written AVX2/NEON kernels: wins over blocked for
//              single-threaded workloads when OMP threads would hurt.
//   blocked -- cache-blocked C++; compiler auto-vectorises inner loops; always
//              faster than naive seq for n beyond L1 cache.
//   seq     -- textbook loops; reference only, never selected automatically.

/// Default backend for dense vector/matrix ops (matmul, matvec, dot, axpy,
/// etc.). Automatically selected at configure time: blas > omp > simd >
/// blocked.
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

/// Best backend for memory-bound vector ops (dot, axpy, scale, norm).
/// Identical to default_backend -- both follow blas > omp > simd > blocked.
inline constexpr Backend best_backend = default_backend;

// Full hierarchy for factorizations, SVD, and eigensolvers:
//   lapack > omp > seq
//
// Rationale:
//   lapack  -- LAPACKE (dgetrf, dgeqrf, dgesdd, dsyevd, etc.):
//   decades-optimised,
//              BLAS-3 internally, fastest for n > ~64.
//   omp     -- our parallel Jacobi eigensolver; meaningful when LAPACK is
//   absent. seq     -- our textbook implementations (Doolittle LU, Householder
//   QR, etc.).
//
// Note: blas and blocked are NOT in this chain -- there is no BLAS-level LU/QR/
// SVD/eigen API (those are LAPACK-level).  If Backend::blas or Backend::blocked
// reaches a factorization dispatcher it silently falls through to seq.

/// Best backend for factorizations, SVD, and eigensolvers.
/// Prefers LAPACKE (industry-standard), then omp, then seq.
inline constexpr Backend lapack_backend =
#if defined(NUMERICS_HAS_LAPACK)
    Backend::lapack;
#elif defined(NUMERICS_HAS_OMP)
    Backend::omp;
#else
    Backend::seq;
#endif

} // namespace num
