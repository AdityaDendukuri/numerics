/// @file lu.hpp
/// @brief LU factorization with partial pivoting.
#pragma once

#include "core/matrix.hpp"
#include "core/policy.hpp"
#include <vector>

namespace num {

/// @brief Packed factorization \f$PA=LU\f$.
struct LUResult {
  Matrix LU;
  std::vector<idx> piv;
  bool singular = false;
};

LUResult lu(const Matrix& A, Backend backend = lapack_backend);

/// @brief Solve \f$Ax=b\f$ from a precomputed \f$PA=LU\f$ factorization.
void lu_solve(const LUResult& f, const Vector& b, Vector& x);

/// @brief Solve \f$AX=B\f$ from a precomputed \f$PA=LU\f$ factorization.
void lu_solve(const LUResult& f, const Matrix& B, Matrix& X);

/// @brief Compute \f$\det(A)=\det(P)^{-1}\prod_i U_{ii}\f$.
real lu_det(const LUResult& f);

/// @brief Compute \f$A^{-1}\f$ by solving \f$AX=I\f$.
Matrix lu_inv(const LUResult& f);

} // namespace num
