/// @file lu.hpp
/// @brief LU factorization with partial pivoting.
#pragma once

#include "core/matrix.hpp"
#include "core/policy.hpp"
#include <vector>

namespace num {

/// @brief Packed factorization \f$PA=LU\f$.
struct LUResult {
    Matrix LU;             ///< Packed unit-lower and upper factors.
    std::vector<idx> piv;  ///< Zero-based row swaps applied during factorization.
    bool singular = false; ///< True when a zero pivot was encountered.
};

/// Factor a square matrix with partial pivoting; singularity is reported in the result.
LUResult lu(const Matrix &A, Backend backend = lapack_backend);

/// @brief Solve \f$Ax=b\f$ from a precomputed \f$PA=LU\f$ factorization.
void lu_solve(const LUResult &f, const Vector &b, Vector &x);

/// @brief Solve \f$AX=B\f$ from a precomputed \f$PA=LU\f$ factorization.
void lu_solve(const LUResult &f, const Matrix &B, Matrix &X);

/// Solve A^T x=b from a precomputed PA=LU factorization.
void lu_solve_transpose(const LUResult &f, const Vector &b, Vector &x);
/// Solve A^T X=B for several right-hand sides.
void lu_solve_transpose(const LUResult &f, const Matrix &B, Matrix &X);

/// Replace one or more right-hand sides with the corresponding solutions.
void solve_in_place(const LUResult &f, Vector &right_hand_side);
void solve_in_place(const LUResult &f, Matrix &right_hand_sides);

/// @brief Compute \f$\det(A)=\det(P)^{-1}\prod_i U_{ii}\f$.
real lu_det(const LUResult &f);

/// @brief Compute \f$A^{-1}\f$ by solving \f$AX=I\f$.
Matrix lu_inv(const LUResult &f);

} // namespace num
