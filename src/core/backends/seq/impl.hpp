/// @file core/backends/seq/impl.hpp
/// @brief Private declarations for the sequential backend.
/// Only included by src/core/vector.cpp and src/core/matrix.cpp.
#pragma once
#include "core/vector.hpp"
#include "core/matrix.hpp"

namespace num::backends::seq {

void scale(Vector& v, real alpha);
void add(const Vector& x, const Vector& y, Vector& z);
void axpy(real alpha, const Vector& x, Vector& y);
real dot(const Vector& x, const Vector& y);
real norm(const Vector& x);

void matmul(const Matrix& A, const Matrix& B, Matrix& C);
void matmul_blocked(const Matrix& A,
                    const Matrix& B,
                    Matrix&       C,
                    idx           block_size);
void matmul_register_blocked(const Matrix& A,
                             const Matrix& B,
                             Matrix&       C,
                             idx           block_size,
                             idx           reg_size);
void matvec(const Matrix& A, const Vector& x, Vector& y);
void matadd(real alpha, const Matrix& A, real beta, const Matrix& B, Matrix& C);

} // namespace num::backends::seq
