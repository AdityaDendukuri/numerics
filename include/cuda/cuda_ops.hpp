/// @file cuda_ops.hpp
/// @brief CUDA kernel wrappers
#pragma once


#include "core/types.hpp"

namespace num::cuda {

/// @brief Allocate device memory
real *alloc(idx n);

/// @brief Free device memory
void free(real *ptr);

/// @brief Copy host to device
void to_device(real *dst, const real *src, idx n);

/// @brief Copy device to host
void to_host(real *dst, const real *src, idx n);

/// @brief v = alpha * v
void scale(real *v, idx n, real alpha);

/// @brief z = x + y
void add(const real *x, const real *y, real *z, idx n);

/// @brief y = alpha*x + y
void axpy(real alpha, const real *x, real *y, idx n);

/// @brief dot product
real dot(const real *x, const real *y, idx n);

/// @brief y = A * x (row-major A)
void matvec(const real *A, const real *x, real *y, idx rows, idx cols);

/// @brief C = A * B
void matmul(const real *A, const real *B, real *C, idx m, idx k, idx n);

/// @brief Batched Thomas algorithm for tridiagonal systems
/// @param a  Lower diagonals (batch_size arrays of size n-1, packed
/// consecutively)
/// @param b  Main diagonals (batch_size arrays of size n)
/// @param c  Upper diagonals (batch_size arrays of size n-1, packed
/// consecutively)
/// @param d  Right-hand sides (batch_size arrays of size n)
/// @param x  Solution vectors (batch_size arrays of size n)
/// @param n  Size of each system
/// @param batch_size  Number of independent systems to solve
void thomas_batched(const real *a, const real *b, const real *c, const real *d, real *x, idx n,
                    idx batch_size);

#if !defined(NUMERICS_HAS_CUDA)

// Without a device build these are no-ops, so host code compiles and links
// unchanged. The GPU backend tags fall back to seq in the same situation.
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
