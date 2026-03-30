/// @file core/vector.cpp
/// @brief Backend dispatch for real-vector ops, and sequential complex-vector ops.
///
/// BasicVector<T> member functions are defined inline in vector.hpp (template).
/// This file only provides:
///   1. Backend-dispatched free functions for Vector (= BasicVector<real>)
///   2. Sequential free functions for CVector (= BasicVector<cplx>)
///
/// Adding a new backend:
///   1. Add the enumerator to enum class Backend in include/core/policy.hpp
///   2. Create src/core/backends/<name>/ with impl.hpp and vector.cpp
///   3. Add `case Backend::<name>:` to each switch below
///   4. Register the .cpp in cmake/sources.cmake

#include "core/vector.hpp"
#include <cmath>

#include "backends/seq/impl.hpp"
#include "backends/blas/impl.hpp"
#include "backends/omp/impl.hpp"
#include "backends/gpu/impl.hpp"

namespace num {

// -- Real-vector dispatch ------------------------------------------------------

void scale(Vector& v, real alpha, Backend b) {
    switch (b) {
    case Backend::seq:
    case Backend::blocked:
    case Backend::simd:     backends::seq::scale(v, alpha);  break;
    case Backend::lapack:   [[fallthrough]];
    case Backend::blas:     backends::blas::scale(v, alpha); break;
    case Backend::omp:      backends::omp::scale(v, alpha);  break;
    case Backend::gpu:      backends::gpu::scale(v, alpha);  break;
    }
}

void add(const Vector& x, const Vector& y, Vector& z, Backend b) {
    if (b == Backend::gpu) {
        cuda::add(x.gpu_data(), y.gpu_data(), z.gpu_data(), x.size());
    } else {
        backends::seq::add(x, y, z);
    }
}

void axpy(real alpha, const Vector& x, Vector& y, Backend b) {
    switch (b) {
    case Backend::seq:
    case Backend::blocked:
    case Backend::simd:     backends::seq::axpy(alpha, x, y);  break;
    case Backend::lapack:   [[fallthrough]];
    case Backend::blas:     backends::blas::axpy(alpha, x, y); break;
    case Backend::omp:      backends::omp::axpy(alpha, x, y);  break;
    case Backend::gpu:      backends::gpu::axpy(alpha, x, y);  break;
    }
}

real dot(const Vector& x, const Vector& y, Backend b) {
    switch (b) {
    case Backend::seq:
    case Backend::blocked:
    case Backend::simd:     return backends::seq::dot(x, y);
    case Backend::lapack:   [[fallthrough]];
    case Backend::blas:     return backends::blas::dot(x, y);
    case Backend::omp:      return backends::omp::dot(x, y);
    case Backend::gpu:      return backends::gpu::dot(x, y);
    }
    return backends::seq::dot(x, y);
}

real norm(const Vector& x, Backend b) {
    switch (b) {
    case Backend::seq:
    case Backend::blocked:
    case Backend::simd:     return backends::seq::norm(x);
    case Backend::lapack:   [[fallthrough]];
    case Backend::blas:     return backends::blas::norm(x);
    case Backend::omp:      return backends::seq::norm(x);   // no OMP norm
    case Backend::gpu:      return backends::gpu::norm(x);
    }
    return backends::seq::norm(x);
}

// -- Complex-vector (sequential) -----------------------------------------------

void scale(CVector& v, cplx alpha) {
    for (idx i = 0; i < v.size(); ++i) v[i] *= alpha;
}

void axpy(cplx alpha, const CVector& x, CVector& y) {
    for (idx i = 0; i < x.size(); ++i) y[i] += alpha * x[i];
}

cplx dot(const CVector& x, const CVector& y) {
    cplx sum{0, 0};
    for (idx i = 0; i < x.size(); ++i) sum += std::conj(x[i]) * y[i];
    return sum;
}

real norm(const CVector& x) {
    real sum = 0;
    for (idx i = 0; i < x.size(); ++i) sum += std::norm(x[i]);
    return std::sqrt(sum);
}

} // namespace num
