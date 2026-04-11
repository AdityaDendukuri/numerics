/// @file core/backends/gpu/matrix.cpp
/// @brief GPU (CUDA) backend  -- matrix operations
///
/// Thin shims that forward to cuda:: kernels (cuda_ops.hpp).
/// Falls back to sequential when NUMERICS_HAS_CUDA is not defined.

#include "core/matrix.hpp"
#include "core/parallel/cuda_ops.hpp"
#include "../seq/impl.hpp"

namespace num::backends::gpu {

void matmul(const Matrix& A, const Matrix& B, Matrix& C) {
#ifdef NUMERICS_HAS_CUDA
    cuda::matmul(A.gpu_data(),
                 B.gpu_data(),
                 C.gpu_data(),
                 A.rows(),
                 A.cols(),
                 B.cols());
#else
    num::backends::seq::matmul(A, B, C);
#endif
}

void matvec(const Matrix& A, const Vector& x, Vector& y) {
#ifdef NUMERICS_HAS_CUDA
    cuda::matvec(A.gpu_data(), x.gpu_data(), y.gpu_data(), A.rows(), A.cols());
#else
    num::backends::seq::matvec(A, x, y);
#endif
}

} // namespace num::backends::gpu
