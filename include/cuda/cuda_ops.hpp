/// @file cuda/cuda_ops.hpp
/// @brief CUDA device kernels over raw pointers.
#pragma once


#include "core/types.hpp"

namespace num::cuda {

/// @brief Allocate `n` elements on the GPU.
real *alloc(idx n);

/// @brief Free memory on the GPU.
void free(real *ptr);

/// @brief Copy from the host to the GPU.
void to_device(real *dst, const real *src, idx n);

/// @brief Copy from the GPU to the host.
void to_host(real *dst, const real *src, idx n);

/// @brief Scale a vector: \f$v \leftarrow \alpha v\f$.
void scale(real *v, idx n, real alpha);

/// @brief Add two vectors: \f$z \leftarrow x + y\f$.
void add(const real *x, const real *y, real *z, idx n);

/// @brief Add a multiple of one vector to another (AXPY): \f$y \leftarrow \alpha x + y\f$.
void axpy(real alpha, const real *x, real *y, idx n);

/// @brief Compute the dot product: \f$x^T y\f$.
real dot(const real *x, const real *y, idx n);

/// @brief Multiply a row-major matrix and a vector: \f$y \leftarrow A x\f$.
void matvec(const real *A, const real *x, real *y, idx rows, idx cols);

/// @brief Multiply two matrices: \f$C \leftarrow A B\f$.
void matmul(const real *A, const real *B, real *C, idx m, idx k, idx n);

/// @brief Solve `batch_size` tridiagonal systems by the Thomas algorithm: \f$T_b x_b = d_b\f$.
/// Diagonals and right-hand sides are packed consecutively per system.
void thomas_batched(const real *a, const real *b, const real *c, const real *d, real *x, idx n,
                    idx batch_size);

#if !defined(NUMERICS_HAS_CUDA)

// Without a device build these are no-ops, so host code compiles and links
// unchanged. The GPU backend tags fall back to seq in the same situation.
/// @brief Raise an error from a GPU routine in a build without CUDA.
[[noreturn]] static void no_cuda() {
    throw std::runtime_error("CUDA not available");
}

inline real *alloc(idx) {
    no_cuda();
}
inline void free(real *) {
    no_cuda();
}
inline void to_device(real *, const real *, idx) {
    no_cuda();
}
inline void to_host(real *, const real *, idx) {
    no_cuda();
}
inline void scale(real *, idx, real) {
    no_cuda();
}
inline void add(const real *, const real *, real *, idx) {
    no_cuda();
}
inline void axpy(real, const real *, real *, idx) {
    no_cuda();
}
inline real dot(const real *, const real *, idx) {
    no_cuda();
}
inline void matvec(const real *, const real *, real *, idx, idx) {
    no_cuda();
}
inline void matmul(const real *, const real *, real *, idx, idx, idx) {
    no_cuda();
}
inline void thomas_batched(const real *, const real *, const real *, const real *, real *, idx, idx) {
    no_cuda();
}

#endif

} // namespace num::cuda
