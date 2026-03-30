/// @file cuda_ops.hpp
/// @brief CUDA kernel wrappers
#pragma once

#include "core/types.hpp"

namespace num::cuda {

/// @brief Allocate device memory
real* alloc(idx n);

/// @brief Free device memory
void free(real* ptr);

/// @brief Copy host to device
void to_device(real* dst, const real* src, idx n);

/// @brief Copy device to host
void to_host(real* dst, const real* src, idx n);

/// @brief v = alpha * v
void scale(real* v, idx n, real alpha);

/// @brief z = x + y
void add(const real* x, const real* y, real* z, idx n);

/// @brief y = alpha*x + y
void axpy(real alpha, const real* x, real* y, idx n);

/// @brief dot product
real dot(const real* x, const real* y, idx n);

/// @brief y = A * x (row-major A)
void matvec(const real* A, const real* x, real* y, idx rows, idx cols);

/// @brief C = A * B
void matmul(const real* A, const real* B, real* C, idx m, idx k, idx n);

/// @brief Batched Thomas algorithm for tridiagonal systems
/// @param a  Lower diagonals (batch_size arrays of size n-1, packed consecutively)
/// @param b  Main diagonals (batch_size arrays of size n)
/// @param c  Upper diagonals (batch_size arrays of size n-1, packed consecutively)
/// @param d  Right-hand sides (batch_size arrays of size n)
/// @param x  Solution vectors (batch_size arrays of size n)
/// @param n  Size of each system
/// @param batch_size  Number of independent systems to solve
void thomas_batched(const real* a, const real* b, const real* c,
                    const real* d, real* x, idx n, idx batch_size);

} // namespace num::cuda
