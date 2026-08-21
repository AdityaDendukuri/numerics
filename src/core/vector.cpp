/// @file core/vector.cpp
/// @brief Vector operations and multi-backend execution dispatch.

#include "core/vector.hpp"
#include "kernel/raw.hpp"
#include <cmath>
#include <cstdio>

#ifdef NUMERICS_HAS_BLAS
    #include <cblas.h>
#endif

#ifdef NUMERICS_HAS_CUDA
    #include "core/parallel/cuda_ops.hpp"
#endif

namespace {
void warn_blas_unavailable() {
#ifndef NUMERICS_HAS_BLAS
    static bool warned = false;
    if (!warned) {
        warned = true;
        std::fprintf(stderr,
                     "[numerics] WARNING: Backend::blas requested but BLAS was not found at configure time.\n"
                     "           Falling back to Backend::seq.\n");
    }
#endif
}
} // namespace

namespace num {

namespace backends {

namespace seq {
void scale(Vector& v, real alpha) {
    kernel::raw::scale(v.data(), alpha, v.size());
}
void add(const Vector& x, const Vector& y, Vector& z) {
    for (idx i = 0; i < x.size(); ++i) {
        z[i] = x[i] + y[i];
}
}
void axpy(real alpha, const Vector& x, Vector& y) {
    kernel::raw::axpy(y.data(), x.data(), alpha, x.size());
}
real dot(const Vector& x, const Vector& y) {
    return kernel::raw::dot(x.data(), y.data(), x.size());
}
real norm(const Vector& x) {
    return kernel::raw::norm(x.data(), x.size());
}
} // namespace seq

namespace blas {
void scale(Vector& v, real alpha) {
    warn_blas_unavailable();
#ifdef NUMERICS_HAS_BLAS
    cblas_dscal(static_cast<int>(v.size()), alpha, v.data(), 1);
#else
    seq::scale(v, alpha);
#endif
}
void axpy(real alpha, const Vector& x, Vector& y) {
    warn_blas_unavailable();
#ifdef NUMERICS_HAS_BLAS
    cblas_daxpy(static_cast<int>(x.size()), alpha, x.data(), 1, y.data(), 1);
#else
    seq::axpy(alpha, x, y);
#endif
}
real dot(const Vector& x, const Vector& y) {
    warn_blas_unavailable();
#ifdef NUMERICS_HAS_BLAS
    return cblas_ddot(static_cast<int>(x.size()), x.data(), 1, y.data(), 1);
#else
    return seq::dot(x, y);
#endif
}
real norm(const Vector& x) {
    warn_blas_unavailable();
#ifdef NUMERICS_HAS_BLAS
    return cblas_dnrm2(static_cast<int>(x.size()), x.data(), 1);
#else
    return seq::norm(x);
#endif
}
} // namespace blas

namespace omp {
void scale(Vector& v, real alpha) {
#ifdef NUMERICS_HAS_OMP
    const idx n = v.size();
    #pragma omp parallel for schedule(static)
    for (idx i = 0; i < n; ++i) {
        v[i] *= alpha;
}
#else
    seq::scale(v, alpha);
#endif
}
void axpy(real alpha, const Vector& x, Vector& y) {
#ifdef NUMERICS_HAS_OMP
    const idx n = x.size();
    #pragma omp parallel for schedule(static)
    for (idx i = 0; i < n; ++i) {
        y[i] += alpha * x[i];
}
#else
    seq::axpy(alpha, x, y);
#endif
}
real dot(const Vector& x, const Vector& y) {
#ifdef NUMERICS_HAS_OMP
    real sum = 0;
    const idx n = x.size();
    #pragma omp parallel for reduction(+ : sum) schedule(static)
    for (idx i = 0; i < n; ++i) {
        sum += x[i] * y[i];
}
    return sum;
#else
    return seq::dot(x, y);
#endif
}
} // namespace omp

namespace gpu {
void scale(Vector& v, real alpha) {
#ifdef NUMERICS_HAS_CUDA
    cuda::scale(v.gpu_data(), v.size(), alpha);
#else
    seq::scale(v, alpha);
#endif
}
void axpy(real alpha, const Vector& x, Vector& y) {
#ifdef NUMERICS_HAS_CUDA
    cuda::axpy(alpha, x.gpu_data(), y.gpu_data(), x.size());
#else
    seq::axpy(alpha, x, y);
#endif
}
real dot(const Vector& x, const Vector& y) {
#ifdef NUMERICS_HAS_CUDA
    return cuda::dot(x.gpu_data(), y.gpu_data(), x.size());
#else
    return seq::dot(x, y);
#endif
}
real norm(const Vector& x) {
#ifdef NUMERICS_HAS_CUDA
    real d = cuda::dot(x.gpu_data(), x.gpu_data(), x.size());
    return std::sqrt(d);
#else
    return seq::norm(x);
#endif
}
} // namespace gpu

} // namespace backends

// Public dispatch functions

void scale(Vector& v, real alpha, Backend b) {
    switch (b) {
        case Backend::seq:
        case Backend::blocked:
        case Backend::simd:
            backends::seq::scale(v, alpha);
            break;
        case Backend::lapack:
            [[fallthrough]];
        case Backend::blas:
            backends::blas::scale(v, alpha);
            break;
        case Backend::omp:
            backends::omp::scale(v, alpha);
            break;
        case Backend::gpu:
            backends::gpu::scale(v, alpha);
            break;
    }
}

void add(const Vector& x, const Vector& y, Vector& z, Backend b) {
    if (b == Backend::gpu) {
#ifdef NUMERICS_HAS_CUDA
        cuda::add(x.gpu_data(), y.gpu_data(), z.gpu_data(), x.size());
#else
        backends::seq::add(x, y, z);
#endif
    } else {
        backends::seq::add(x, y, z);
    }
}

void axpy(real alpha, const Vector& x, Vector& y, Backend b) {
    switch (b) {
        case Backend::seq:
        case Backend::blocked:
        case Backend::simd:
            backends::seq::axpy(alpha, x, y);
            break;
        case Backend::lapack:
            [[fallthrough]];
        case Backend::blas:
            backends::blas::axpy(alpha, x, y);
            break;
        case Backend::omp:
            backends::omp::axpy(alpha, x, y);
            break;
        case Backend::gpu:
            backends::gpu::axpy(alpha, x, y);
            break;
    }
}

real dot(const Vector& x, const Vector& y, Backend b) {
    switch (b) {
        case Backend::seq:
        case Backend::blocked:
        case Backend::simd:
            return backends::seq::dot(x, y);
        case Backend::lapack:
            [[fallthrough]];
        case Backend::blas:
            return backends::blas::dot(x, y);
        case Backend::omp:
            return backends::omp::dot(x, y);
        case Backend::gpu:
            return backends::gpu::dot(x, y);
    }
    return backends::seq::dot(x, y);
}

real norm(const Vector& x, Backend b) {
    switch (b) {
        case Backend::seq:
        case Backend::blocked:
        case Backend::simd:
            return backends::seq::norm(x);
        case Backend::lapack:
            [[fallthrough]];
        case Backend::blas:
            return backends::blas::norm(x);
        case Backend::omp:
            return backends::seq::norm(x);
        case Backend::gpu:
            return backends::gpu::norm(x);
    }
    return backends::seq::norm(x);
}

void scale(CVector& v, cplx alpha) {
    for (idx i = 0; i < v.size(); ++i) {
        v[i] *= alpha;
}
}

void axpy(cplx alpha, const CVector& x, CVector& y) {
    for (idx i = 0; i < x.size(); ++i) {
        y[i] += alpha * x[i];
}
}

cplx dot(const CVector& x, const CVector& y) {
    cplx sum{0, 0};
    for (idx i = 0; i < x.size(); ++i) {
        sum += std::conj(x[i]) * y[i];
}
    return sum;
}

real norm(const CVector& x) {
    real sum = 0;
    for (idx i = 0; i < x.size(); ++i) {
        sum += std::norm(x[i]);
}
    return std::sqrt(sum);
}

template class BasicVector<double>;

} // namespace num
