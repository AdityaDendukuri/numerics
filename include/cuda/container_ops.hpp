/// @file cuda/container_ops.hpp
/// @brief CUDA operations on `vec<real>`.
#pragma once

#include "container/matrix.hpp"
#include "container/vector.hpp"
#include "cuda/cuda_ops.hpp"
#include <cmath>

namespace num::cuda {

inline void scale(vec<real> &v, real alpha) noexcept { scale(v.gpu_data(), v.size(), alpha); }

inline void axpy(real alpha, const vec<real> &x, vec<real> &y) noexcept {
    axpy(alpha, x.gpu_data(), y.gpu_data(), x.size());
}

[[nodiscard]] inline real dot(const vec<real> &x, const vec<real> &y) noexcept {
    return dot(x.gpu_data(), y.gpu_data(), x.size());
}

[[nodiscard]] inline real norm(const vec<real> &x) noexcept { return std::sqrt(dot(x, x)); }

inline void add(const vec<real> &x, const vec<real> &y, vec<real> &z) noexcept {
    add(x.gpu_data(), y.gpu_data(), z.gpu_data(), x.size());
}

inline void matvec(const mat<real> &A, const vec<real> &x, vec<real> &y) {
    matvec(A.gpu_data(), x.gpu_data(), y.gpu_data(), A.rows(), A.cols());
}

inline void matmul(const mat<real> &A, const mat<real> &B, mat<real> &C) {
    matmul(A.gpu_data(), B.gpu_data(), C.gpu_data(), A.rows(), A.cols(), B.cols());
}

} // namespace num::cuda
