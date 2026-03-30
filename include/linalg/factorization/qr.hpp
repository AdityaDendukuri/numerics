/// @file qr.hpp
/// @brief QR factorization via Householder reflections
#pragma once

#include "core/matrix.hpp"
#include "core/policy.hpp"

namespace num {

/// @brief Result of a QR factorization: A = Q * R
///
/// Q is an mxm orthogonal matrix (Q^T * Q = I).
/// R is an mxn upper triangular matrix (entries below the diagonal are zero).
///
/// For an overdetermined system (m > n), the least-squares solution minimises
/// ||A*x - b||_2 and is obtained by back-substituting into R[:n,:n] * x = (Q^T*b)[:n].
/// The residual norm is ||(Q^T*b)[n:]||_2.
struct QRResult {
    Matrix Q;   ///< mxm orthogonal
    Matrix R;   ///< mxn upper triangular
};

/// @brief QR factorization of an mxn matrix A (m >= n) via Householder reflections.
///
/// Each Householder step k:
///   1. Extract column k of the current working matrix from row k downward.
///   2. Compute v = x + sign(x[0]) * ||x|| * e_1  (sign chosen to avoid cancellation).
///   3. Normalise v.
///   4. Apply H_k = I - 2*v*v^T to the trailing submatrix.
///
/// After min(m-1, n) steps, the matrix is upper triangular.
/// Q is reconstructed by accumulating the reflectors in reverse order.
///
/// @param backend  Backend::lapack uses LAPACKE_dgeqrf (default when available).
///                 Backend::seq    uses our Householder implementation.
QRResult qr(const Matrix& A, Backend backend = lapack_backend);

/// @brief Solve the least-squares problem  min ||A*x - b||_2.
///
/// Algorithm:
///   1. y = Q^T * b
///   2. Back-substitute: solve R[:n,:n] * x = y[:n]
///
/// For square non-singular A this returns the exact solution.
/// For overdetermined A (m > n) this returns the minimum-residual solution.
void qr_solve(const QRResult& f, const Vector& b, Vector& x);

} // namespace num
