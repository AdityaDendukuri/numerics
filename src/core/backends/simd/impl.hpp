/// @file core/backends/simd/impl.hpp
/// @brief Private declarations for the SIMD backend.
/// Only included by src/core/vector.cpp and src/core/matrix.cpp.
#pragma once
#include "core/vector.hpp"
#include "core/matrix.hpp"

namespace num::backends::simd {

void matmul(const Matrix& A, const Matrix& B, Matrix& C, idx block_size);
void matvec(const Matrix& A, const Vector& x, Vector& y);

} // namespace num::backends::simd
