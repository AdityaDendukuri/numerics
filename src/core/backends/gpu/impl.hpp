/// @file core/backends/gpu/impl.hpp
/// @brief Private declarations for the GPU (CUDA) backend.
/// Only included by src/core/vector.cpp and src/core/matrix.cpp.
#pragma once
#include "core/vector.hpp"
#include "core/matrix.hpp"

namespace num::backends::gpu {

void scale(Vector& v, real alpha);
void axpy(real alpha, const Vector& x, Vector& y);
real dot(const Vector& x, const Vector& y);
real norm(const Vector& x);

void matmul(const Matrix& A, const Matrix& B, Matrix& C);
void matvec(const Matrix& A, const Vector& x, Vector& y);

} // namespace num::backends::gpu
