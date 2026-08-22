/// @file qr.hpp
/// @brief QR factorization via Householder reflections.
#pragma once

#include "core/matrix.hpp"
#include "core/policy.hpp"

namespace num {

/// @brief QR factorization \f$A=QR\f$.
struct QRResult {
    Matrix Q; ///< Orthonormal factor.
    Matrix R; ///< Upper-triangular factor.
};

/// @brief Factor \f$A\in\mathbb{R}^{m\times n}\f$ as \f$A=QR\f$.
QRResult qr(const Matrix &A, Backend backend = lapack_backend);

/// @brief Solve \f$\min_x \|Ax-b\|_2\f$.
void qr_solve(const QRResult &f, const Vector &b, Vector &x);

} // namespace num
