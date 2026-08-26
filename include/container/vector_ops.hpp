/// @file container/vector_ops.hpp
/// @brief Level-1 operations on dense vectors, dispatched at compile time.
///
/// Each routine has one overload per backend tag. Selection therefore happens
/// during compilation and the body inlines into the caller: a dot product inside
/// a CG iteration is a loop, not a call through a switch in another translation
/// unit. A tag whose backend was not detected falls back to `seq`.
#pragma once

#include "container/vector.hpp"
#include "core/policy.hpp"
#include "kernel/raw.hpp"
#include <cmath>
#include <span>
#include <stdexcept>

#if defined(NUMERICS_HAS_BLAS)
#include <cblas.h>
#endif
#if defined(NUMERICS_HAS_CUDA)
#include "container/parallel/cuda_ops.hpp"
#endif

namespace num {

// -- scale: v <- alpha v -------------------------------------------------------

inline void scale(Vector &v, real alpha, backend::seq_t) noexcept {
    kernel::raw::scale(v.data(), alpha, v.size());
}

inline void scale(Vector &v, real alpha, backend::blocked_t) noexcept {
    scale(v, alpha, backend::seq);
}

inline void scale(Vector &v, real alpha, backend::simd_t) noexcept {
    scale(v, alpha, backend::seq);
}

inline void scale(Vector &v, real alpha, backend::blas_t) noexcept {
#if defined(NUMERICS_HAS_BLAS)
    cblas_dscal(static_cast<int>(v.size()), alpha, v.data(), 1);
#else
    scale(v, alpha, backend::seq);
#endif
}

inline void scale(Vector &v, real alpha, backend::lapack_t) noexcept {
    scale(v, alpha, backend::blas);
}

inline void scale(Vector &v, real alpha, backend::omp_t) noexcept {
#if defined(NUMERICS_HAS_OMP)
    const idx n = v.size();
    real *d = v.data();
#pragma omp parallel for schedule(static)
    for (idx i = 0; i < n; ++i) {
        d[i] *= alpha;
    }
#else
    scale(v, alpha, backend::seq);
#endif
}

inline void scale(Vector &v, real alpha, backend::gpu_t) noexcept {
#if defined(NUMERICS_HAS_CUDA)
    cuda::scale(v.gpu_data(), v.size(), alpha);
#else
    scale(v, alpha, backend::seq);
#endif
}

inline void scale(Vector &v, real alpha) noexcept {
    scale(v, alpha, backend::dflt);
}

// -- axpy: y <- y + alpha x ----------------------------------------------------

inline void axpy(real alpha, const Vector &x, Vector &y, backend::seq_t) noexcept {
    kernel::raw::axpy(y.data(), x.data(), alpha, x.size());
}

inline void axpy(real alpha, const Vector &x, Vector &y, backend::blocked_t) noexcept {
    axpy(alpha, x, y, backend::seq);
}

inline void axpy(real alpha, const Vector &x, Vector &y, backend::simd_t) noexcept {
    axpy(alpha, x, y, backend::seq);
}

inline void axpy(real alpha, const Vector &x, Vector &y, backend::blas_t) noexcept {
#if defined(NUMERICS_HAS_BLAS)
    cblas_daxpy(static_cast<int>(x.size()), alpha, x.data(), 1, y.data(), 1);
#else
    axpy(alpha, x, y, backend::seq);
#endif
}

inline void axpy(real alpha, const Vector &x, Vector &y, backend::lapack_t) noexcept {
    axpy(alpha, x, y, backend::blas);
}

inline void axpy(real alpha, const Vector &x, Vector &y, backend::omp_t) noexcept {
#if defined(NUMERICS_HAS_OMP)
    const idx n = x.size();
    const real *xd = x.data();
    real *yd = y.data();
#pragma omp parallel for schedule(static)
    for (idx i = 0; i < n; ++i) {
        yd[i] += alpha * xd[i];
    }
#else
    axpy(alpha, x, y, backend::seq);
#endif
}

inline void axpy(real alpha, const Vector &x, Vector &y, backend::gpu_t) noexcept {
#if defined(NUMERICS_HAS_CUDA)
    cuda::axpy(alpha, x.gpu_data(), y.gpu_data(), x.size());
#else
    axpy(alpha, x, y, backend::seq);
#endif
}

inline void axpy(real alpha, const Vector &x, Vector &y) noexcept {
    axpy(alpha, x, y, backend::dflt);
}

// -- dot: x^T y ----------------------------------------------------------------

[[nodiscard]] inline real dot(const Vector &x, const Vector &y, backend::seq_t) noexcept {
    return kernel::raw::dot(x.data(), y.data(), x.size());
}

[[nodiscard]] inline real dot(const Vector &x, const Vector &y, backend::blocked_t) noexcept {
    return dot(x, y, backend::seq);
}

[[nodiscard]] inline real dot(const Vector &x, const Vector &y, backend::simd_t) noexcept {
    return dot(x, y, backend::seq);
}

[[nodiscard]] inline real dot(const Vector &x, const Vector &y, backend::blas_t) noexcept {
#if defined(NUMERICS_HAS_BLAS)
    return cblas_ddot(static_cast<int>(x.size()), x.data(), 1, y.data(), 1);
#else
    return dot(x, y, backend::seq);
#endif
}

[[nodiscard]] inline real dot(const Vector &x, const Vector &y, backend::lapack_t) noexcept {
    return dot(x, y, backend::blas);
}

[[nodiscard]] inline real dot(const Vector &x, const Vector &y, backend::omp_t) noexcept {
#if defined(NUMERICS_HAS_OMP)
    real sum = 0.0;
    const idx n = x.size();
    const real *xd = x.data();
    const real *yd = y.data();
#pragma omp parallel for reduction(+ : sum) schedule(static)
    for (idx i = 0; i < n; ++i) {
        sum += xd[i] * yd[i];
    }
    return sum;
#else
    return dot(x, y, backend::seq);
#endif
}

[[nodiscard]] inline real dot(const Vector &x, const Vector &y, backend::gpu_t) noexcept {
#if defined(NUMERICS_HAS_CUDA)
    return cuda::dot(x.gpu_data(), y.gpu_data(), x.size());
#else
    return dot(x, y, backend::seq);
#endif
}

[[nodiscard]] inline real dot(const Vector &x, const Vector &y) noexcept {
    return dot(x, y, backend::dflt);
}

/// @brief Sequential dot product over non-owning spans.
///
/// Views are not vector spaces (there is nowhere for a sum to live), but an inner
/// product only reads, so this is well defined on them.
[[nodiscard]] inline real dot(std::span<const real> x, std::span<const real> y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("dot: vector sizes must match");
    }
    return kernel::raw::dot(x.data(), y.data(), x.size());
}

// -- norm: ||x||_2 -------------------------------------------------------------

[[nodiscard]] inline real norm(const Vector &x, backend::seq_t) noexcept {
    return kernel::raw::norm(x.data(), x.size());
}

[[nodiscard]] inline real norm(const Vector &x, backend::blocked_t) noexcept {
    return norm(x, backend::seq);
}

[[nodiscard]] inline real norm(const Vector &x, backend::simd_t) noexcept {
    return norm(x, backend::seq);
}

[[nodiscard]] inline real norm(const Vector &x, backend::blas_t) noexcept {
#if defined(NUMERICS_HAS_BLAS)
    return cblas_dnrm2(static_cast<int>(x.size()), x.data(), 1);
#else
    return norm(x, backend::seq);
#endif
}

[[nodiscard]] inline real norm(const Vector &x, backend::lapack_t) noexcept {
    return norm(x, backend::blas);
}

[[nodiscard]] inline real norm(const Vector &x, backend::omp_t) noexcept {
    return std::sqrt(dot(x, x, backend::omp));
}

[[nodiscard]] inline real norm(const Vector &x, backend::gpu_t) noexcept {
#if defined(NUMERICS_HAS_CUDA)
    return std::sqrt(cuda::dot(x.gpu_data(), x.gpu_data(), x.size()));
#else
    return norm(x, backend::seq);
#endif
}

[[nodiscard]] inline real norm(const Vector &x) noexcept {
    return norm(x, backend::dflt);
}

// -- add: z = x + y ------------------------------------------------------------

inline void add(const Vector &x, const Vector &y, Vector &z, backend::seq_t) noexcept {
    const idx n = x.size();
    for (idx i = 0; i < n; ++i) {
        z[i] = x[i] + y[i];
    }
}

template <class Tag>
inline void add(const Vector &x, const Vector &y, Vector &z, Tag) noexcept {
    add(x, y, z, backend::seq);
}

inline void add(const Vector &x, const Vector &y, Vector &z, backend::gpu_t) noexcept {
#if defined(NUMERICS_HAS_CUDA)
    cuda::add(x.gpu_data(), y.gpu_data(), z.gpu_data(), x.size());
#else
    add(x, y, z, backend::seq);
#endif
}

inline void add(const Vector &x, const Vector &y, Vector &z) noexcept {
    add(x, y, z, backend::dflt);
}

} // namespace num

namespace num {

// -----------------------------------------------------------------------------
// Runtime bridge
// -----------------------------------------------------------------------------
//
// For a backend value not known until run time. Each converts the value to a tag
// once, then calls the same compile-time overload set above. Prefer passing a tag
// directly: that inlines, whereas this cannot.

inline void scale(Vector &v, real alpha, Backend b) {
    with_backend(b, [&](auto tag) { scale(v, alpha, tag); });
}

inline void axpy(real alpha, const Vector &x, Vector &y, Backend b) {
    with_backend(b, [&](auto tag) { axpy(alpha, x, y, tag); });
}

[[nodiscard]] inline real dot(const Vector &x, const Vector &y, Backend b) {
    return with_backend(b, [&](auto tag) { return dot(x, y, tag); });
}

[[nodiscard]] inline real norm(const Vector &x, Backend b) {
    return with_backend(b, [&](auto tag) { return norm(x, tag); });
}

inline void add(const Vector &x, const Vector &y, Vector &z, Backend b) {
    with_backend(b, [&](auto tag) { add(x, y, z, tag); });
}

// -- complex level-1 -----------------------------------------------------------
//
// Sequential only. BLAS exposes these, but the complex Krylov methods that use
// them are not backend-dispatched, so a tag family here would be unused weight.

inline void scale(CVector &v, cplx alpha) noexcept {
    const idx n = v.size();
    for (idx i = 0; i < n; ++i) {
        v[i] *= alpha;
    }
}

inline void axpy(cplx alpha, const CVector &x, CVector &y) noexcept {
    const idx n = x.size();
    for (idx i = 0; i < n; ++i) {
        y[i] += alpha * x[i];
    }
}

/// @brief Conjugate inner product \f$\langle x,y \rangle = \sum_i \overline{x_i} y_i\f$.
[[nodiscard]] inline cplx dot(const CVector &x, const CVector &y) noexcept {
    cplx sum{};
    const idx n = x.size();
    for (idx i = 0; i < n; ++i) {
        sum += std::conj(x[i]) * y[i];
    }
    return sum;
}

[[nodiscard]] inline real norm(const CVector &x) noexcept {
    real sum = 0.0;
    const idx n = x.size();
    for (idx i = 0; i < n; ++i) {
        sum += std::norm(x[i]);
    }
    return std::sqrt(sum);
}

} // namespace num
