/// @file core/policy.hpp
/// @brief Build capabilities and compile-time backend selection.
///
/// A backend is a *substitution for a leaf kernel*, not a parameter threaded
/// through every algorithm. BLAS replaces roughly six routines (dot, axpy, scal,
/// nrm2, gemv, gemm); LAPACK replaces roughly six factorizations. Everything above
/// them is pure algebra that merely calls those leaves.
///
/// So selection happens once, at the leaf, and at compile time. An algorithm such
/// as CG takes no backend argument: it calls `dot`, and `dot` decides. A runtime
/// enum cannot do this — it forces the body out of line into a translation unit
/// that can `#ifdef` on availability, which is what previously made the algorithms
/// non-inlinable, `double`-only, and impossible to extract.
///
/// `num::Backend` survives for the one case that genuinely needs a runtime choice
/// (a flag read from a config file); see `num::with_backend`.
#pragma once


#include <cstdint>

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
// Backend tags
// -----------------------------------------------------------------------------

/// @brief Runtime backend identifier.
///
/// Retained only for callers that must choose from a value not known until run
/// time. No algorithm signature takes one; use `with_backend` to convert a value
/// into a tag exactly once, at the boundary.
enum class Backend : std::uint8_t {
    seq,
    blocked,
    simd,
    blas,
    omp,
    gpu,
    lapack,
};

/// @brief Compile-time backend selectors.
///
/// Each tag names a strategy for the leaf kernels. Passing one selects an overload,
/// so the choice is resolved during compilation and the body inlines into the
/// caller. A tag whose backend was not detected at configure time falls back to
/// `seq` rather than failing.
namespace backend {

// Each tag converts to its runtime identity. A routine that dispatches at compile
// time takes the tag and inlines; one that has not been converted yet still accepts
// it as a Backend value. That lets the two coexist, so the migration proceeds file
// by file rather than in one step, and a caller writes backend::blas either way.

struct seq_t { ///< Portable scalar loops. Always available.
    constexpr operator Backend() const noexcept { return Backend::seq; }
};
struct blocked_t { ///< Cache-blocked scalar loops.
    constexpr operator Backend() const noexcept { return Backend::blocked; }
};
struct simd_t { ///< Explicit vector intrinsics.
    constexpr operator Backend() const noexcept { return Backend::simd; }
};
struct blas_t { ///< External BLAS.
    constexpr operator Backend() const noexcept { return Backend::blas; }
};
struct omp_t { ///< OpenMP parallel loops.
    constexpr operator Backend() const noexcept { return Backend::omp; }
};
struct lapack_t { ///< External LAPACK.
    constexpr operator Backend() const noexcept { return Backend::lapack; }
};
struct gpu_t { ///< CUDA device kernels.
    constexpr operator Backend() const noexcept { return Backend::gpu; }
};

inline constexpr seq_t seq{};
inline constexpr blocked_t blocked{};
inline constexpr simd_t simd{};
inline constexpr blas_t blas{};
inline constexpr omp_t omp{};
inline constexpr lapack_t lapack{};
inline constexpr gpu_t gpu{};

/// @brief The strategy chosen by the build for level-1 and level-2 work.
using default_t =
#if defined(NUMERICS_HAS_BLAS)
    blas_t;
#elif defined(NUMERICS_HAS_OMP)
    omp_t;
#elif defined(NUMERICS_HAS_SIMD)
    simd_t;
#else
    blocked_t;
#endif

/// @brief The strategy chosen by the build for factorizations.
using factor_t =
#if defined(NUMERICS_HAS_LAPACK)
    lapack_t;
#elif defined(NUMERICS_HAS_OMP)
    omp_t;
#else
    seq_t;
#endif

inline constexpr default_t dflt{};
inline constexpr factor_t factor{};

} // namespace backend

// -----------------------------------------------------------------------------
// Runtime selection
// -----------------------------------------------------------------------------


/// @brief Invoke `f` with the compile-time tag corresponding to a runtime value.
///
/// The single point in the library where a runtime backend value becomes a type.
/// Every strategy is instantiated, so this trades code size for the ability to
/// choose late; algorithms remain free of backend parameters either way.
///
/// @code
/// num::with_backend(cfg.backend, [&](auto tag) { num::matvec(A, x, y, tag); });
/// @endcode
template <class F>
inline decltype(auto) with_backend(Backend b, F &&f) {
    switch (b) {
    case Backend::seq:
        return f(backend::seq);
    case Backend::blocked:
        return f(backend::blocked);
    case Backend::simd:
        return f(backend::simd);
    case Backend::blas:
        return f(backend::blas);
    case Backend::omp:
        return f(backend::omp);
    case Backend::gpu:
        return f(backend::gpu);
    case Backend::lapack:
        return f(backend::lapack);
    }
    return f(backend::seq);
}

} // namespace num

namespace num::kernel {

/// @brief Sequential execution policy tag.
struct seq_t {};

/// @brief Parallel execution policy tag.
struct par_t {};

inline constexpr seq_t kseq{};
inline constexpr par_t kpar{};

#if defined(NUMERICS_HAS_OMP)
using default_policy = par_t;
#else
using default_policy = seq_t;
#endif

inline constexpr default_policy kdefault{};

} // namespace num::kernel
