/// @file omp/vector_ops.hpp
/// @brief OpenMP-accelerated Level-1 vector operations and threaded block reduction.
///
/// Plain functions, no tag/enum indirection: a caller who wants this backend
/// calls `num::omp::scale`/`num::omp::axpy`/... directly, or an algorithm
/// templated on the backend namespace instantiates with `num::omp`.
///
/// Every function here hands each thread a block and calls the matching
/// `num::kernel` routine on it — never a hand-written per-element loop. OMP's
/// job is strictly the division of labor across threads; the arithmetic inside
/// each thread's block is exactly the same inlined, vectorized loop `num::seq`
/// runs, so there is exactly one implementation of "what scale/axpy/etc. does"
/// in the whole tree, and OpenMP only decides how it's sliced up.
#pragma once

#include "container/vector.hpp"
#include "core/types.hpp"
#include "kernel/kernel.hpp"
#include "omp/parallel_ops.hpp"
#include <cmath>

namespace num::omp {

// -----------------------------------------------------------------------------
// Level-1 vector operations
// -----------------------------------------------------------------------------

inline void scale(vec &v, real alpha) noexcept {
    real *d = v.data();
    parallel_apply(v.size(),
                   [d, alpha](idx offset, idx length) { kernel::scale(d + offset, alpha, length); });
}

inline void axpy(real alpha, const vec &x, vec &y) noexcept {
    const real *xd = x.data();
    real *yd = y.data();
    parallel_apply(x.size(), [xd, yd, alpha](idx offset, idx length) {
        kernel::axpy(yd + offset, xd + offset, alpha, length);
    });
}

[[nodiscard]] inline real dot(const vec &x, const vec &y) noexcept {
    const real *xd = x.data();
    const real *yd = y.data();
    // A block each, reduced by the raw kernel, so threading composes with the
    // vector accumulators instead of replacing them with a scalar chain.
    return parallel_reduce<real>(
        x.size(), [xd, yd](idx offset, idx length) { return kernel::dot(xd + offset, yd + offset, length); });
}

[[nodiscard]] inline real norm(const vec &x) noexcept {
    return std::sqrt(dot(x, x));
}

inline void add(const vec &x, const vec &y, vec &z) noexcept {
    const real *xd = x.data();
    const real *yd = y.data();
    real *zd = z.data();
    parallel_apply(x.size(), [xd, yd, zd](idx offset, idx length) {
        kernel::add(zd + offset, xd + offset, yd + offset, length);
    });
}

inline void axpby(real a, const vec &x, real b, vec &y) noexcept {
    const real *xd = x.data();
    real *yd = y.data();
    parallel_apply(x.size(), [xd, yd, a, b](idx offset, idx length) {
        kernel::axpby(yd + offset, xd + offset, a, b, length);
    });
}

inline void axpbyz(real a, const vec &x, real b, const vec &y, vec &z) noexcept {
    const real *xd = x.data();
    const real *yd = y.data();
    real *zd = z.data();
    parallel_apply(x.size(), [xd, yd, zd, a, b](idx offset, idx length) {
        kernel::axpbyz(zd + offset, xd + offset, yd + offset, a, b, length);
    });
}

} // namespace num::omp
