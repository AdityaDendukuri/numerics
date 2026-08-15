/// @file factorization/cholesky.hpp
/// @brief Dense Cholesky factorization for SPD matrices.
/// @todo Add LAPACK dpotrf/dpotrs backend dispatch and batched/block variants.
#pragma once

#include "core/matrix.hpp"
#include "core/vector.hpp"
#include "linalg/matrix_properties.hpp"

namespace num {

/// @brief Lower-triangular factorization \f$A=LL^T\f$.
struct CholeskyResult {
  Matrix L;
  bool success = false;
};

CholeskyResult cholesky(const linalg::SPDMatrix<Matrix>& A);

CholeskyResult cholesky(const Matrix& A);

void cholesky_solve(const CholeskyResult& f, const Vector& b, Vector& x);

} // namespace num
