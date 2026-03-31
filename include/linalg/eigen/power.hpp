/// @file eigen/power.hpp
/// @brief Power iteration, inverse iteration, Rayleigh quotient iteration
///
/// All three methods accept a backend parameter:
///
///   power_iteration(A)                         // default_backend
///   power_iteration(A, tol, max, num::omp)     // OmpBackend  -- parallel
///   matvec power_iteration(A, tol, max, num::blas)    // BlasBackend  -- BLAS
///   matvec
///
/// The backend is forwarded to every matvec, dot, axpy, and norm call inside
/// the iteration.
#pragma once

#include "core/matrix.hpp"
#include "core/vector.hpp"
#include "core/policy.hpp"
#include "linalg/factorization/lu.hpp"
#include <cmath>
#include <stdexcept>

namespace num {

/// @brief Result of a single-eigenvalue iteration
struct PowerResult {
    real   eigenvalue = 0.0;   ///< Converged eigenvalue (Rayleigh quotient)
    Vector eigenvector;        ///< Corresponding unit eigenvector
    idx    iterations = 0;     ///< Iterations performed
    bool   converged  = false; ///< Whether tolerance was met
};

namespace detail {
/// Normalise v in-place; returns the old norm.
inline real normalise(Vector& v) {
    real nrm = 0;
    for (idx i = 0; i < v.size(); ++i)
        nrm += v[i] * v[i];
    nrm = std::sqrt(nrm);
    if (nrm > 1e-300)
        for (idx i = 0; i < v.size(); ++i)
            v[i] /= nrm;
    return nrm;
}
} // namespace detail

/// @brief Power iteration  -- finds the eigenvalue largest in absolute value.
///
/// @param A        Square matrix (need not be symmetric)
/// @param tol      Tolerance on eigenvalue change between iterations
/// @param max_iter Maximum iterations
/// @param backend  Backend forwarded to matvec and dot
PowerResult power_iteration(const Matrix& A,
                            real          tol      = 1e-10,
                            idx           max_iter = 1000,
                            Backend       backend  = default_backend);

/// @brief Inverse iteration  -- finds the eigenvalue closest to a shift sigma.
///
/// Factorizes (A - sigmaI) once then solves repeatedly.
///
/// @param A        Square matrix (symmetric recommended)
/// @param sigma    Shift  -- should be near the target eigenvalue
/// @param tol      Tolerance on eigenvalue change between iterations
/// @param max_iter Maximum iterations
/// @param backend  Backend forwarded to matvec and dot
PowerResult inverse_iteration(const Matrix& A,
                              real          sigma,
                              real          tol      = 1e-10,
                              idx           max_iter = 1000,
                              Backend       backend  = default_backend);

/// @brief Rayleigh quotient iteration  -- cubically convergent.
///
/// Updates the shift sigma = v^T*A*v at every step -> fresh LU each iteration.
///
/// @param A        Symmetric matrix
/// @param x0       Starting vector (determines which eigenvalue is found)
/// @param tol      Tolerance on residual ||A*v - lambda*v||
/// @param max_iter Maximum iterations
/// @param backend  Backend forwarded to matvec, dot, axpy, norm
PowerResult rayleigh_iteration(const Matrix& A,
                               const Vector& x0,
                               real          tol      = 1e-10,
                               idx           max_iter = 50,
                               Backend       backend  = default_backend);

} // namespace num
