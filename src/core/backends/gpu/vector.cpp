/// @file core/backends/gpu/vector.cpp
/// @brief GPU (CUDA) backend  -- vector operations
///
/// Thin shims that forward to cuda:: kernels (cuda_ops.hpp).
/// Falls back to sequential when NUMERICS_HAS_CUDA is not defined.

#include "core/vector.hpp"
#include "core/parallel/cuda_ops.hpp"
#include "../seq/impl.hpp"
#include <cmath>

namespace num::backends::gpu {

void scale(Vector& v, real alpha) {
#ifdef NUMERICS_HAS_CUDA
    cuda::scale(v.gpu_data(), v.size(), alpha);
#else
    num::backends::seq::scale(v, alpha);
#endif
}

void axpy(real alpha, const Vector& x, Vector& y) {
#ifdef NUMERICS_HAS_CUDA
    cuda::axpy(alpha, x.gpu_data(), y.gpu_data(), x.size());
#else
    num::backends::seq::axpy(alpha, x, y);
#endif
}

real dot(const Vector& x, const Vector& y) {
#ifdef NUMERICS_HAS_CUDA
    return cuda::dot(x.gpu_data(), y.gpu_data(), x.size());
#else
    return num::backends::seq::dot(x, y);
#endif
}

real norm(const Vector& x) {
#ifdef NUMERICS_HAS_CUDA
    real d = cuda::dot(x.gpu_data(), x.gpu_data(), x.size());
    return std::sqrt(d);
#else
    return num::backends::seq::norm(x);
#endif
}

} // namespace num::backends::gpu
