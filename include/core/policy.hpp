/// @file core/policy.hpp
/// @brief Build capabilities and the one compile-time backend default.
///
/// A backend is a *substitution for a leaf kernel*, not a parameter threaded
/// through every algorithm. BLAS replaces roughly six routines (dot, axpy, scal,
/// nrm2, gemv, gemm); LAPACK replaces roughly six factorizations. Everything above
/// them is pure algebra that merely calls those leaves.
///
/// Each backend is a plain namespace of free functions matching `num::kernel`'s
/// signatures: `num::kernel` (the portable reference, always available, and the
/// thing every other one funnels back into), `num::seq`, `num::omp`, `num::blas`,
/// `num::cuda`. There is no tag or enum indirection between a caller and these:
/// name the namespace you want. `num::accel` below is the single build-time
/// default, so untagged call sites have something to resolve to.
///
/// A namespace is not a type, so an algorithm cannot be templated on which
/// backend to call. Where one genuinely must run on more than one, it takes a
/// `bool` (e.g. `template <bool Parallel = has_omp>`) and picks inside with
/// `if constexpr`; where it must not, it just names a namespace.
#pragma once


#include "core/types.hpp"
#include <concepts>
#include <cstdint>
#include <type_traits>

namespace num {

// -----------------------------------------------------------------------------
// Build capabilities
// -----------------------------------------------------------------------------

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

// Whether the build enabled a wider instruction set than the target baseline
// (`-mavx2 -mfma` on x86-64; NEON is baseline on AArch64). This is a fact about
// the compiler's flags, not a backend: `num::kernel` has no intrinsics and no
// runtime CPU dispatch, and gets its vectorization from the compiler either way.
// Only the FFT's explicit-intrinsic path reads the underlying macros.
inline constexpr bool has_simd =
#if defined(NUMERICS_HAS_SIMD)
    true;
#else
    false;
#endif

inline constexpr bool has_cuda =
#if defined(NUMERICS_HAS_CUDA)
    true;
#else
    false;
#endif

// -----------------------------------------------------------------------------
// The one compile-time default
// -----------------------------------------------------------------------------

// Forward-declare every backend namespace so `accel` can name whichever is
// available; each backend's own headers (include/<name>/*.hpp) reopen these
// with real content. Reopening an empty namespace declaration is ordinary C++,
// the same way num::kernel itself is built up across several headers.
//
// `num::seq` is the fallback when no accelerator was configured: `num::kernel`
// itself cannot be `accel`'s target because it only knows raw pointers, not
// `vec`/`mat` — `num::seq` (defined in container/vector_ops.hpp and
// container/matrix_ops.hpp, where those types are visible) is the thin
// container-aware wrapper around `num::kernel` that plays that role.
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
