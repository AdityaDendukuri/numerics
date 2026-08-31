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

template <class Tag = backend::default_t>
inline void scale(Vector &v, real alpha, Tag = {}) noexcept {
    if constexpr (std::is_same_v<Tag, backend::blas_t> && has_blas) {
#if defined(NUMERICS_HAS_BLAS)
        cblas_dscal(static_cast<int>(v.size()), alpha, v.data(), 1);
#else
        kernel::raw::scale(v.data(), alpha, v.size());
#endif
    } else if constexpr (std::is_same_v<Tag, backend::omp_t> && has_omp) {
#if defined(NUMERICS_HAS_OMP)
        const idx n = v.size();
        real *d = v.data();
#pragma omp parallel for schedule(static)
        for (idx i = 0; i < n; ++i) {
            d[i] *= alpha;
        }
#else
        kernel::raw::scale(v.data(), alpha, v.size());
#endif
    } else if constexpr (std::is_same_v<Tag, backend::gpu_t> && has_cuda) {
#if defined(NUMERICS_HAS_CUDA)
        cuda::scale(v.gpu_data(), v.size(), alpha);
#else
        kernel::raw::scale(v.data(), alpha, v.size());
#endif
    } else {
        kernel::raw::scale(v.data(), alpha, v.size());
    }
}

// -- axpy: y <- y + alpha x ----------------------------------------------------

template <class Tag = backend::default_t>
inline void axpy(real alpha, const Vector &x, Vector &y, Tag = {}) noexcept {
    if constexpr (std::is_same_v<Tag, backend::blas_t> && has_blas) {
#if defined(NUMERICS_HAS_BLAS)
        cblas_daxpy(static_cast<int>(x.size()), alpha, x.data(), 1, y.data(), 1);
#else
        kernel::raw::axpy(y.data(), x.data(), alpha, x.size());
#endif
    } else if constexpr (std::is_same_v<Tag, backend::omp_t> && has_omp) {
#if defined(NUMERICS_HAS_OMP)
        const idx n = x.size();
        const real *xd = x.data();
        real *yd = y.data();
#pragma omp parallel for schedule(static)
        for (idx i = 0; i < n; ++i) {
            yd[i] += alpha * xd[i];
        }
#else
        kernel::raw::axpy(y.data(), x.data(), alpha, x.size());
#endif
    } else if constexpr (std::is_same_v<Tag, backend::gpu_t> && has_cuda) {
#if defined(NUMERICS_HAS_CUDA)
        cuda::axpy(alpha, x.gpu_data(), y.gpu_data(), x.size());
#else
        kernel::raw::axpy(y.data(), x.data(), alpha, x.size());
#endif
    } else {
        kernel::raw::axpy(y.data(), x.data(), alpha, x.size());
    }
}

// -- dot: x^T y ----------------------------------------------------------------

template <class Tag = backend::default_t>
[[nodiscard]] inline real dot(const Vector &x, const Vector &y, Tag = {}) noexcept {
    if constexpr (std::is_same_v<Tag, backend::blas_t> && has_blas) {
#if defined(NUMERICS_HAS_BLAS)
        return cblas_ddot(static_cast<int>(x.size()), x.data(), 1, y.data(), 1);
#else
        return kernel::raw::dot(x.data(), y.data(), x.size());
#endif
    } else if constexpr (std::is_same_v<Tag, backend::omp_t> && has_omp) {
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
        return kernel::raw::dot(x.data(), y.data(), x.size());
#endif
    } else if constexpr (std::is_same_v<Tag, backend::gpu_t> && has_cuda) {
#if defined(NUMERICS_HAS_CUDA)
        return cuda::dot(x.gpu_data(), y.gpu_data(), x.size());
#else
        return kernel::raw::dot(x.data(), y.data(), x.size());
#endif
    } else {
        return kernel::raw::dot(x.data(), y.data(), x.size());
    }
}

/// @brief Sequential dot product over non-owning spans.
[[nodiscard]] inline real dot(std::span<const real> x, std::span<const real> y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("dot: vector sizes must match");
    }
    return kernel::raw::dot(x.data(), y.data(), x.size());
}

// -- norm: ||x||_2 -------------------------------------------------------------

template <class Tag = backend::default_t>
[[nodiscard]] inline real norm(const Vector &x, Tag tag = {}) noexcept {
    if constexpr (std::is_same_v<Tag, backend::blas_t> && has_blas) {
#if defined(NUMERICS_HAS_BLAS)
        return cblas_dnrm2(static_cast<int>(x.size()), x.data(), 1);
#else
        return kernel::raw::norm(x.data(), x.size());
#endif
    } else if constexpr (std::is_same_v<Tag, backend::omp_t> || std::is_same_v<Tag, backend::gpu_t>) {
        return std::sqrt(dot(x, x, tag));
    } else {
        return kernel::raw::norm(x.data(), x.size());
    }
}

// -- add: z = x + y ------------------------------------------------------------

template <class Tag = backend::default_t>
inline void add(const Vector &x, const Vector &y, Vector &z, Tag = {}) noexcept {
    if constexpr (std::is_same_v<Tag, backend::gpu_t> && has_cuda) {
#if defined(NUMERICS_HAS_CUDA)
        cuda::add(x.gpu_data(), y.gpu_data(), z.gpu_data(), x.size());
#else
        const idx n = x.size();
        for (idx i = 0; i < n; ++i) {
            z[i] = x[i] + y[i];
        }
#endif
    } else {
        const idx n = x.size();
        for (idx i = 0; i < n; ++i) {
            z[i] = x[i] + y[i];
        }
    }
}

// -----------------------------------------------------------------------------
// Runtime bridge
// -----------------------------------------------------------------------------

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
