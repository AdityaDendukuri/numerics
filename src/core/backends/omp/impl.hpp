/// @file core/backends/omp/impl.hpp
/// @brief Private declarations for the OpenMP backend.
/// Only included by src/core/vector.cpp and src/core/matrix.cpp.
#pragma once
#include "core/vector.hpp"
#include "core/matrix.hpp"

namespace num::backends::omp {

void scale(Vector& v, real alpha);
void axpy(real alpha, const Vector& x, Vector& y);
real dot(const Vector& x, const Vector& y);

void matmul(const Matrix& A, const Matrix& B, Matrix& C);
void matvec(const Matrix& A, const Vector& x, Vector& y);
void matadd(real alpha, const Matrix& A, real beta, const Matrix& B, Matrix& C);

} // namespace num::backends::omp
