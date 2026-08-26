/// @file linear/eigen/power.hpp
/// @brief Power iteration, inverse iteration, Rayleigh quotient iteration
///
/// All three methods accept a backend parameter:
///
///   power_iteration(A)                         // backend::dflt
///   power_iteration(A, tol, max, num::omp)     // OmpBackend  -- parallel
///   matvec power_iteration(A, tol, max, num::blas)    // BlasBackend  -- BLAS
///   matvec
///
/// The backend is forwarded to every matvec, dot, axpy, and norm call inside
/// the iteration.
#pragma once

#include "container/matrix.hpp"
#include "container/matrix_ops.hpp"
#include "container/vector.hpp"
#include "container/vector_ops.hpp"
#include "core/policy.hpp"
#include "linear/factorization/lu.hpp"
#include <cmath>
#include <stdexcept>

namespace num {

/// @brief Result of a single-eigenvalue iteration
struct PowerResult {
    real eigenvalue = 0.0;  ///< Converged eigenvalue (Rayleigh quotient)
    Vector eigenvector;     ///< Corresponding unit eigenvector
    idx iterations = 0;     ///< Iterations performed
    bool converged = false; ///< Whether tolerance was met
};

namespace detail {
/// Normalise v in-place; returns the old norm.
inline real normalise(Vector &v) {
    // nrm <- ||v||_2
    const real nrm = kernel::raw::norm(v.data(), v.size());
    if (nrm > 1e-300) {
        // v <- v/||v||_2
        kernel::raw::scale(v.data(), real(1) / nrm, v.size());
    }
    return nrm;
}
} // namespace detail

/// @brief Power iteration  -- finds the eigenvalue largest in absolute value.
///
/// @param A        Square matrix (need not be symmetric)
/// @param tol      Tolerance on eigenvalue change between iterations
/// @param max_iter Maximum iterations
/// @param backend  Backend forwarded to matvec and dot
PowerResult power_iteration(const Matrix &A, real tol = 1e-10, idx max_iter = 1000,
                            Backend backend = backend::dflt);

/// @brief Inverse iteration  -- finds the eigenvalue closest to a shift sigma.
///
/// Factorizes (A - sigmaI) once then solves repeatedly.
///
/// @param A        Square matrix (symmetric recommended)
/// @param sigma    Shift  -- should be near the target eigenvalue
/// @param tol      Tolerance on eigenvalue change between iterations
/// @param max_iter Maximum iterations
/// @param backend  Backend forwarded to matvec and dot
PowerResult inverse_iteration(const Matrix &A, real sigma, real tol = 1e-10, idx max_iter = 1000,
                              Backend backend = backend::dflt);

/// @brief Rayleigh quotient iteration  -- cubically convergent.
///
/// Updates the shift sigma = v^T*A*v at every step -> fresh LU each iteration.
///
/// @param A        Symmetric matrix
/// @param x0       Starting vector (determines which eigenvalue is found)
/// @param tol      Tolerance on residual ||A*v - lambda*v||
/// @param max_iter Maximum iterations
/// @param backend  Backend forwarded to matvec, dot, axpy, norm
PowerResult rayleigh_iteration(const Matrix &A, const Vector &x0, real tol = 1e-10,
                               idx max_iter = 50, Backend backend = backend::dflt);

inline PowerResult power_iteration(const Matrix &A, real tol, idx max_iter, Backend backend) {
    constexpr real tiny = 1e-300;
    const idx n = A.rows();
    if (A.cols() != n) {
        throw std::invalid_argument("power_iteration: matrix must be square");
    }

    Vector v(n, 0.0);
    v[0] = 1.0;

    real lambda = 0.0;
    PowerResult result{0.0, v, 0, false};

    for (idx iter = 0; iter < max_iter; ++iter) {
        result.iterations = iter + 1;

        Vector w(n);
        matvec(A, v, w, backend);

        real new_lambda = dot(v, w, backend);
        detail::normalise(w);

        real delta = std::abs(new_lambda - lambda);
        lambda = new_lambda;
        v = w;

        if (delta < tol * (std::abs(lambda) + tiny)) {
            result.converged = true;
            break;
        }
    }

    result.eigenvalue = lambda;
    result.eigenvector = v;
    return result;
}

inline PowerResult inverse_iteration(const Matrix &A, real sigma, real tol, idx max_iter,
                                     Backend backend) {
    constexpr real tiny = 1e-300;
    const idx n = A.rows();
    if (A.cols() != n) {
        throw std::invalid_argument("inverse_iteration: matrix must be square");
    }

    // Factorize (A - sigma*I) once
    Matrix M = A;
    for (idx i = 0; i < n; ++i) {
        M(i, i) -= sigma;
    }
    // M is A (rejected above unless square) shifted along its diagonal.
    LUResult f = lu(assume_square(M));

    Vector v(n, 0.0);
    v[0] = 1.0;

    real lambda = 0.0;
    PowerResult result{0.0, v, 0, false};

    for (idx iter = 0; iter < max_iter; ++iter) {
        result.iterations = iter + 1;

        Vector w(n);
        lu_solve(f, v, w);
        detail::normalise(w);

        // Rayleigh quotient as eigenvalue estimate
        Vector av(n);
        matvec(A, w, av, backend);
        real new_lambda = dot(w, av, backend);

        real delta = std::abs(new_lambda - lambda);
        lambda = new_lambda;
        v = w;

        if (delta < tol * (std::abs(lambda) + tiny)) {
            result.converged = true;
            break;
        }
    }

    result.eigenvalue = lambda;
    result.eigenvector = v;
    return result;
}

inline PowerResult rayleigh_iteration(const Matrix &A, const Vector &x0, real tol, idx max_iter,
                                      Backend backend) {
    const idx n = A.rows();
    if (A.cols() != n) {
        throw std::invalid_argument("rayleigh_iteration: matrix must be square");
    }
    if (x0.size() != n) {
        throw std::invalid_argument("rayleigh_iteration: x0 size mismatch");
    }

    Vector v = x0;
    detail::normalise(v);

    // Initial Rayleigh quotient
    Vector av(n);
    matvec(A, v, av, backend);
    real sigma = dot(v, av, backend);

    PowerResult result{sigma, v, 0, false};

    for (idx iter = 0; iter < max_iter; ++iter) {
        result.iterations = iter + 1;

        // Factorize (A - sigma*I); fresh each iteration (cubic convergence)
        Matrix M = A;
        for (idx i = 0; i < n; ++i) {
            M(i, i) -= sigma;
        }
        // M is A (rejected above unless square) shifted along its diagonal.
        LUResult f = lu(assume_square(M));

        if (f.singular) {
            break;
        }

        Vector w(n);
        lu_solve(f, v, w);
        detail::normalise(w);

        matvec(A, w, av, backend);
        real new_sigma = dot(w, av, backend);

        // res <- ||A*w - sigma*w||_2
        const real res = std::sqrt(
            kernel::raw::linear_combination_norm_sq(av.data(), real(1), w.data(), -new_sigma, n));

        sigma = new_sigma;
        v = w;

        if (res < tol) {
            result.converged = true;
            break;
        }
    }

    result.eigenvalue = sigma;
    result.eigenvector = v;
    return result;
}

} // namespace num
