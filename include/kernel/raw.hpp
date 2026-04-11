/// @file kernel/raw.hpp
/// @brief Tier-1 kernel: raw-pointer, inline, zero-overhead inner loops.
///
/// These are the lowest-level building blocks in the numerics stack.
/// Every higher tier (type-safe wrappers, operator interface, solvers)
/// ultimately reduces to these loops.
///
/// Design invariants:
///   - All functions are NUM_K_AINLINE: the compiler always sees the loop body
///     and can vectorize, fuse, or eliminate it during inlining.
///   - NUM_K_RESTRICT on every pointer: tells the compiler inputs do not alias,
///     which is necessary for auto-vectorization of most loops.
///   - noexcept throughout: no exception machinery in hot paths.
///   - No heap allocation: callers own all memory.
///   - BLAS dispatch (#ifdef NUMERICS_HAS_BLAS): routes to cblas Level-1/2
///     when available. BLAS handles its own internal threading and is safe
///     to call from any context (no nested-OMP issue).
///   - No OpenMP: OMP at raw level causes nested-parallelism problems when
///     a higher-level kernel wraps these in its own parallel region.
///     OMP parallelism lives at Tier 2.
///
/// Operations:
///   --- Level 1 (vector) ---
///   scale(x, alpha, n)          x[i] *= alpha
///   axpy(y, x, alpha, n)        y[i] += alpha * x[i]
///   axpby(y, x, a, b, n)        y[i] = a*x[i] + b*y[i]      [fused, no BLAS eq.]
///   axpbyz(z, x, y, a, b, n)    z[i] = a*x[i] + b*y[i]      [fused, no BLAS eq.]
///   dot(x, y, n)                return sum x[i]*y[i]
///   norm_sq(x, n)               return sum x[i]^2            [no BLAS sqrt cost]
///   norm(x, n)                  return sqrt(norm_sq)
///   l1_norm(x, n)               return sum |x[i]|
///   linf_norm(x, n)             return max |x[i]|
///   sum(x, n)                   return sum x[i]              [no BLAS eq.]
///
///   --- Level 2 (matrix-vector, row-major) ---
///   matvec(y, A, x, m, n)       y[i] = sum_j A[i*n+j] * x[j]
///   ger(A, x, y, alpha, m, n)   A[i*n+j] += alpha * x[i] * y[j]
///   trsv_lower(x, L, b, n)      solve Lx = b  (L lower triangular)
///   trsv_upper(x, U, b, n)      solve Ux = b  (U upper triangular)
///
/// Include this header when writing a fused custom kernel that needs to
/// compose multiple operations into a single memory pass.
/// For ordinary user code prefer num::kernel:: (Tier 2).
#pragma once

#include "core/types.hpp"
#include <cmath>
#include <cstring>

#ifdef NUMERICS_HAS_BLAS
#  include <cblas.h>
#endif

// ---------------------------------------------------------------------------
// Portability annotations
//
//   NUM_K_AINLINE  -- force-inline (compiler always sees the loop body)
//   NUM_K_RESTRICT -- no-alias pointer hint (enables auto-vectorization)
//   NUM_K_IVDEP    -- assert no loop-carried dependencies to the vectorizer
//
// All three are no-ops on unsupported compilers; the code still compiles,
// just without the extra hints.
// ---------------------------------------------------------------------------

#if defined(__GNUC__) || defined(__clang__)
#  define NUM_K_AINLINE  [[gnu::always_inline]] inline
#  define NUM_K_RESTRICT __restrict__
#  define NUM_K_IVDEP    _Pragma("GCC ivdep")
#else
#  define NUM_K_AINLINE  inline
#  define NUM_K_RESTRICT
#  define NUM_K_IVDEP
#endif

namespace num::kernel::raw {

// ===========================================================================
// Level 1: vector operations
// ===========================================================================

/// @brief x[i] *= alpha
NUM_K_AINLINE
void scale(real* NUM_K_RESTRICT x, real alpha, idx n) noexcept {
#ifdef NUMERICS_HAS_BLAS
    cblas_dscal(static_cast<int>(n), alpha, x, 1);
#else
    NUM_K_IVDEP
    for (idx i = 0; i < n; ++i) {
        x[i] *= alpha;
    }
#endif
}

/// @brief y[i] += alpha * x[i]
NUM_K_AINLINE
void axpy(real* NUM_K_RESTRICT y, const real* NUM_K_RESTRICT x,
          real alpha, idx n) noexcept {
#ifdef NUMERICS_HAS_BLAS
    cblas_daxpy(static_cast<int>(n), alpha, x, 1, y, 1);
#else
    NUM_K_IVDEP
    for (idx i = 0; i < n; ++i) {
        y[i] += alpha * x[i];
    }
#endif
}

/// @brief y[i] = a*x[i] + b*y[i]  (fused scale-and-add, one memory pass)
///
/// No direct BLAS equivalent. Avoids a separate scale pass then axpy pass:
/// compared to scale(y,b); axpy(a,x,y), this reads y only once.
NUM_K_AINLINE
void axpby(real* NUM_K_RESTRICT y, const real* NUM_K_RESTRICT x,
           real a, real b, idx n) noexcept {
    NUM_K_IVDEP
    for (idx i = 0; i < n; ++i) {
        y[i] = (a * x[i]) + (b * y[i]);
    }
}

/// @brief z[i] = a*x[i] + b*y[i]  (fused, three-array, one pass each)
///
/// No direct BLAS equivalent. z may alias x or y; otherwise fully independent.
NUM_K_AINLINE
void axpbyz(real* NUM_K_RESTRICT z,
            const real* NUM_K_RESTRICT x,
            const real* NUM_K_RESTRICT y,
            real a, real b, idx n) noexcept {
    NUM_K_IVDEP
    for (idx i = 0; i < n; ++i) {
        z[i] = (a * x[i]) + (b * y[i]);
    }
}

/// @brief dot product: return sum x[i] * y[i]
NUM_K_AINLINE
real dot(const real* NUM_K_RESTRICT x, const real* NUM_K_RESTRICT y,
         idx n) noexcept {
#ifdef NUMERICS_HAS_BLAS
    return cblas_ddot(static_cast<int>(n), x, 1, y, 1);
#else
    real s = real(0);
    NUM_K_IVDEP
    for (idx i = 0; i < n; ++i) {
        s += x[i] * y[i];
    }
    return s;
#endif
}

/// @brief sum x[i]^2  (no sqrt; use for convergence checks to avoid sqrt cost)
[[nodiscard]] NUM_K_AINLINE
real norm_sq(const real* NUM_K_RESTRICT x, idx n) noexcept {
    real s = real(0);
    NUM_K_IVDEP
    for (idx i = 0; i < n; ++i) {
        s += x[i] * x[i];
    }
    return s;
}

/// @brief Euclidean norm: sqrt(sum x[i]^2)
[[nodiscard]] NUM_K_AINLINE
real norm(const real* NUM_K_RESTRICT x, idx n) noexcept {
#ifdef NUMERICS_HAS_BLAS
    return cblas_dnrm2(static_cast<int>(n), x, 1);
#else
    return std::sqrt(norm_sq(x, n));
#endif
}

/// @brief L1 norm: sum |x[i]|
[[nodiscard]] NUM_K_AINLINE
real l1_norm(const real* NUM_K_RESTRICT x, idx n) noexcept {
#ifdef NUMERICS_HAS_BLAS
    return cblas_dasum(static_cast<int>(n), x, 1);
#else
    real s = real(0);
    NUM_K_IVDEP
    for (idx i = 0; i < n; ++i) {
        s += std::abs(x[i]);
    }
    return s;
#endif
}

/// @brief L-infinity norm: max |x[i]|
[[nodiscard]] NUM_K_AINLINE
real linf_norm(const real* NUM_K_RESTRICT x, idx n) noexcept {
#ifdef NUMERICS_HAS_BLAS
    // cblas_idamax returns the 0-based index of the element with max |value|
    const int k = cblas_idamax(static_cast<int>(n), x, 1);
    return std::abs(x[k]);
#else
    real mx = real(0);
    for (idx i = 0; i < n; ++i) {
        const real v = std::abs(x[i]);
        if (v > mx) {
            mx = v;
        }
    }
    return mx;
#endif
}

/// @brief Scalar sum: return sum x[i]
///
/// No BLAS equivalent (cblas_dasum sums absolute values; this does not).
[[nodiscard]] NUM_K_AINLINE
real sum(const real* NUM_K_RESTRICT x, idx n) noexcept {
    real s = real(0);
    NUM_K_IVDEP
    for (idx i = 0; i < n; ++i) {
        s += x[i];
    }
    return s;
}

// ===========================================================================
// Level 2: matrix-vector operations  (row-major storage throughout)
// ===========================================================================

/// @brief y[i] = sum_j A[i*n + j] * x[j]  (m x n row-major matrix)
///
/// y must be pre-allocated with size m. y and x must not alias A.
NUM_K_AINLINE
void matvec(real* NUM_K_RESTRICT y,
            const real* NUM_K_RESTRICT A,
            const real* NUM_K_RESTRICT x,
            idx m, idx n) noexcept {
#ifdef NUMERICS_HAS_BLAS
    cblas_dgemv(CblasRowMajor, CblasNoTrans,
                static_cast<int>(m), static_cast<int>(n),
                1.0, A, static_cast<int>(n),
                x, 1,
                0.0, y, 1);
#else
    for (idx i = 0; i < m; ++i) {
        real s = real(0);
        const real* row = A + (i * n);
        NUM_K_IVDEP
        for (idx j = 0; j < n; ++j) {
            s += row[j] * x[j];
        }
        y[i] = s;
    }
#endif
}

/// @brief Rank-1 update: A[i*n + j] += alpha * x[i] * y[j]  (m x n row-major)
///
/// BLAS dger equivalent. Inner loop over j is independent and vectorizable.
NUM_K_AINLINE
void ger(real* NUM_K_RESTRICT A,
         const real* NUM_K_RESTRICT x,
         const real* NUM_K_RESTRICT y,
         real alpha, idx m, idx n) noexcept {
#ifdef NUMERICS_HAS_BLAS
    cblas_dger(CblasRowMajor,
               static_cast<int>(m), static_cast<int>(n),
               alpha, x, 1, y, 1,
               A, static_cast<int>(n));
#else
    for (idx i = 0; i < m; ++i) {
        real* NUM_K_RESTRICT row = A + (i * n);
        const real axi = alpha * x[i];
        NUM_K_IVDEP
        for (idx j = 0; j < n; ++j) {
            row[j] += axi * y[j];
        }
    }
#endif
}

/// @brief Forward substitution: solve Lx = b, L lower triangular (n x n, row-major).
///
/// x must be pre-allocated with size n. b and x must not alias each other.
/// Uses BLAS dtrsv when available (copies b -> x first, then solves in-place).
NUM_K_AINLINE
void trsv_lower(real* NUM_K_RESTRICT x,
                const real* NUM_K_RESTRICT L,
                const real* NUM_K_RESTRICT b,
                idx n) noexcept {
#ifdef NUMERICS_HAS_BLAS
    std::memcpy(x, b, n * sizeof(real));
    cblas_dtrsv(CblasRowMajor, CblasLower, CblasNoTrans, CblasNonUnit,
                static_cast<int>(n), L, static_cast<int>(n), x, 1);
#else
    for (idx i = 0; i < n; ++i) {
        real s = b[i];
        const real* row = L + i * n;
        for (idx j = 0; j < i; ++j) {
            s -= row[j] * x[j];
        }
        x[i] = s / row[i];
    }
#endif
}

/// @brief Back substitution: solve Ux = b, U upper triangular (n x n, row-major).
///
/// x must be pre-allocated with size n. b and x must not alias each other.
/// Uses BLAS dtrsv when available (copies b -> x first, then solves in-place).
NUM_K_AINLINE
void trsv_upper(real* NUM_K_RESTRICT x,
                const real* NUM_K_RESTRICT U,
                const real* NUM_K_RESTRICT b,
                idx n) noexcept {
#ifdef NUMERICS_HAS_BLAS
    std::memcpy(x, b, n * sizeof(real));
    cblas_dtrsv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit,
                static_cast<int>(n), U, static_cast<int>(n), x, 1);
#else
    for (idx i = n; i-- > 0;) {
        real s = b[i];
        const real* row = U + i * n;
        for (idx j = i + 1; j < n; ++j) {
            s -= row[j] * x[j];
        }
        x[i] = s / row[i];
    }
#endif
}

} // namespace num::kernel::raw
