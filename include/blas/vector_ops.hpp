/// @file blas/vector_ops.hpp
/// @brief BLAS-accelerated Level-1 vector operations.
#pragma once

#include "container/vector.hpp"
#include "core/types.hpp"
#include "kernel/kernel.hpp"
#include <cmath>
#include <cstdio>

#if defined(NUMERICS_HAS_BLAS)
#include <cblas.h>
#endif

namespace num::blas {

/// One-shot notice when a BLAS path is requested but BLAS was not configured.
inline void warn_unavailable() {
#if !defined(NUMERICS_HAS_BLAS)
    static bool warned = false;
    if (!warned) {
        warned = true;
        std::fprintf(stderr,
                     "[numerics] WARNING: num::blas requested but BLAS was not found at "
                     "configure time.\n           Falling back to num::kernel.\n");
    }
#endif
}

inline void scale(vec &v, real alpha) noexcept {
#if defined(NUMERICS_HAS_BLAS)
    cblas_dscal(static_cast<int>(v.size()), alpha, v.data(), 1);
#else
    warn_unavailable();
    kernel::scale(v.data(), alpha, v.size());
#endif
}

inline void axpy(real alpha, const vec &x, vec &y) noexcept {
#if defined(NUMERICS_HAS_BLAS)
    cblas_daxpy(static_cast<int>(x.size()), alpha, x.data(), 1, y.data(), 1);
#else
    warn_unavailable();
    kernel::axpy(y.data(), x.data(), alpha, x.size());
#endif
}

[[nodiscard]] inline real dot(const vec &x, const vec &y) noexcept {
#if defined(NUMERICS_HAS_BLAS)
    return cblas_ddot(static_cast<int>(x.size()), x.data(), 1, y.data(), 1);
#else
    warn_unavailable();
    return kernel::dot(x.data(), y.data(), x.size());
#endif
}

[[nodiscard]] inline real norm(const vec &x) noexcept {
#if defined(NUMERICS_HAS_BLAS)
    return cblas_dnrm2(static_cast<int>(x.size()), x.data(), 1);
#else
    warn_unavailable();
    return kernel::norm(x.data(), x.size());
#endif
}

/// @brief `z <- x + y`. No single BLAS call does an out-of-place add; a plain
/// loop is exactly as fast as `dcopy`+`daxpy` and skips the extra pass.
inline void add(const vec &x, const vec &y, vec &z) noexcept {
    const idx n = x.size();
    const real *xd = x.data();
    const real *yd = y.data();
    real *zd = z.data();
    for (idx i = 0; i < n; ++i) {
        zd[i] = xd[i] + yd[i];
    }
}

} // namespace num::blas
