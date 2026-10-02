/// @file container/vector_ops.hpp
/// @brief Untagged Level-1 vector operations: resolve through `num::accel`.
///
/// A caller who wants a specific backend calls it by name — `num::omp::dot`,
/// `num::blas::axpy`, `num::cuda::scale` — directly; there is no tag or enum
/// layer between the caller and the namespace. These untagged overloads exist
/// only for call sites that do not care which backend runs, and simply forward
/// to whichever `num::accel` resolved to at configure time.
#pragma once

#include "core/types.hpp"
#include "container/vector.hpp"
#include "core/policy.hpp"
#include "kernel/kernel.hpp"
#include <cmath>
#include <complex>
#include <span>
#include <stdexcept>

// `blas` and `omp` are included unconditionally. Each degrades to `num::kernel`
// internally when its library was not configured (see `blas::warn_unavailable`), so the
// namespace and its functions exist in every build — which is what lets a call site write
// `num::blas::dot(x, y)` without an `#ifdef` around it, as the docs promise. CUDA is the
// deliberate exception: it throws rather than silently running on the CPU, and its header
// needs a device toolkit, so it stays gated.
#include "blas/vector_ops.hpp"
#include "omp/vector_ops.hpp"
#if defined(NUMERICS_HAS_CUDA)
#include "cuda/container_ops.hpp"
#endif

namespace num::seq {

/// @brief Thin vec-aware wrappers over `num::kernel`, used when no
/// accelerator (BLAS/OMP/CUDA) was configured. `num::kernel` itself cannot
/// serve this role: it only knows raw pointers, never `vec<real>`.
inline void scale(vec<real> &v, real alpha) noexcept { kernel::scale(v.data(), alpha, v.size()); }

inline void axpy(real alpha, const vec<real> &x, vec<real> &y) noexcept {
    kernel::axpy(y.data(), x.data(), alpha, x.size());
}

[[nodiscard]] inline real dot(const vec<real> &x, const vec<real> &y) noexcept {
    return kernel::dot(x.data(), y.data(), x.size());
}

[[nodiscard]] inline real norm(const vec<real> &x) noexcept { return kernel::norm(x.data(), x.size()); }

inline void add(const vec<real> &x, const vec<real> &y, vec<real> &z) noexcept {
    const idx n = x.size();
    for (idx i = 0; i < n; ++i) {
        z[i] = x[i] + y[i];
    }
}

/// @brief Computes scaled vector update \f$y_i \leftarrow a x_i + b y_i\f$.
inline void axpby(real a, const vec<real> &x, real b, vec<real> &y) noexcept {
    kernel::axpby(y.data(), x.data(), a, b, x.size());
}

/// @brief Computes linear combination \f$z_i \leftarrow a x_i + b y_i\f$.
inline void axpbyz(real a, const vec<real> &x, real b, const vec<real> &y, vec<real> &z) noexcept {
    kernel::axpbyz(z.data(), x.data(), y.data(), a, b, x.size());
}

} // namespace num::seq

namespace num {

/// @brief \f$v \leftarrow \alpha v\f$, through the build's default backend.
inline void scale(vec<real> &v, real alpha) noexcept { accel::scale(v, alpha); }

/// @brief \f$y \leftarrow y + \alpha x\f$, through the build's default backend.
inline void axpy(real alpha, const vec<real> &x, vec<real> &y) noexcept { accel::axpy(alpha, x, y); }

[[nodiscard]] inline real dot(const vec<real> &x, const vec<real> &y) noexcept { return accel::dot(x, y); }

/// @brief Sequential dot product over non-owning spans.
[[nodiscard]] inline real dot(view<const real> x, view<const real> y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("dot: vector sizes must match");
    }
    return kernel::dot(x.data(), y.data(), x.size());
}

/// @brief The Euclidean norm \f$\|x\|_2\f$, through the build's default backend.
[[nodiscard]] inline real norm(const vec<real> &x) noexcept { return accel::norm(x); }

inline void add(const vec<real> &x, const vec<real> &y, vec<real> &z) noexcept { accel::add(x, y, z); }

/// @brief Computes scaled vector update \f$y_i \leftarrow a x_i + b y_i\f$.
inline void axpby(real a, const vec<real> &x, real b, vec<real> &y) noexcept {
#if defined(NUMERICS_HAS_OMP)
    omp::axpby(a, x, b, y);
#else
    seq::axpby(a, x, b, y);
#endif
}

/// @brief Computes linear combination \f$z_i \leftarrow a x_i + b y_i\f$.
inline void axpbyz(real a, const vec<real> &x, real b, const vec<real> &y, vec<real> &z) noexcept {
#if defined(NUMERICS_HAS_OMP)
    omp::axpbyz(a, x, b, y, z);
#else
    seq::axpbyz(a, x, b, y, z);
#endif
}

// -- complex level-1 -----------------------------------------------------------

inline void scale(vec<cplx> &v, cplx alpha) noexcept {
    const idx n = v.size();
    for (idx i = 0; i < n; ++i) {
        v[i] *= alpha;
    }
}

inline void axpy(cplx alpha, const vec<cplx> &x, vec<cplx> &y) noexcept {
    const idx n = x.size();
    for (idx i = 0; i < n; ++i) {
        y[i] += alpha * x[i];
    }
}

[[nodiscard]] inline cplx dot(const vec<cplx> &x, const vec<cplx> &y) noexcept {
    cplx sum{};
    const idx n = x.size();
    for (idx i = 0; i < n; ++i) {
        sum += std::conj(x[i]) * y[i];
    }
    return sum;
}

[[nodiscard]] inline real norm(const vec<cplx> &x) noexcept {
    real sum = 0.0;
    const idx n = x.size();
    for (idx i = 0; i < n; ++i) {
        sum += std::norm(x[i]);
    }
    return std::sqrt(sum);
}

} // namespace num
