/// @file cuda/container_ops.hpp
/// @brief `vec`-level convenience overloads over the raw CUDA device kernels.
///
/// `cuda_ops.hpp` stays raw-pointer-only (device pointers, explicit lengths) so
/// callers that manage device buffers directly — `unsafe::cg`, batched solvers —
/// have nothing above them. These overloads exist only so `num::cuda` can serve
/// as `num::accel` on the same footing as `num::omp`/`num::blas`/`num::seq`.
#pragma once

#include "container/matrix.hpp"
#include "container/vector.hpp"
#include "cuda/cuda_ops.hpp"
#include <cmath>

namespace num::cuda {

inline void scale(vec &v, real alpha) noexcept { scale(v.gpu_data(), v.size(), alpha); }

inline void axpy(real alpha, const vec &x, vec &y) noexcept {
    axpy(alpha, x.gpu_data(), y.gpu_data(), x.size());
}

[[nodiscard]] inline real dot(const vec &x, const vec &y) noexcept {
    return dot(x.gpu_data(), y.gpu_data(), x.size());
}

[[nodiscard]] inline real norm(const vec &x) noexcept { return std::sqrt(dot(x, x)); }

inline void add(const vec &x, const vec &y, vec &z) noexcept {
    add(x.gpu_data(), y.gpu_data(), z.gpu_data(), x.size());
}

inline void matvec(const mat &A, const vec &x, vec &y) {
    matvec(A.gpu_data(), x.gpu_data(), y.gpu_data(), A.rows(), A.cols());
}

inline void matmul(const mat &A, const mat &B, mat &C) {
    matmul(A.gpu_data(), B.gpu_data(), C.gpu_data(), A.rows(), A.cols(), B.cols());
}

} // namespace num::cuda
