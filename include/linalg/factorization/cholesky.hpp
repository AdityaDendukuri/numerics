/// @file factorization/cholesky.hpp
/// @brief Dense Cholesky factorization for SPD matrices.
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

/// @brief Solve \f$AX=B\f$ for several right-hand sides at once.
void cholesky_solve(const CholeskyResult& f, const Matrix& B, Matrix& X);

void solve_in_place(const CholeskyResult& f, Vector& right_hand_side);
void solve_in_place(const CholeskyResult& f, Matrix& right_hand_sides);

/// Replace A=LL^T by A+x*x^T in O(n^2).
void cholesky_update(CholeskyResult& factor, const Vector& update);

/// Replace A=LL^T by A-x*x^T in O(n^2), or throw if it is not SPD.
void cholesky_downdate(CholeskyResult& factor, const Vector& update);

} // namespace num
