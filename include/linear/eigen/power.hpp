/// @file linear/eigen/power.hpp
/// @brief Power iteration, inverse iteration, Rayleigh quotient iteration
///
/// Every matvec/dot/axpy/norm call inside these iterations goes through the
/// untagged `num::matvec`/`num::dot`/... entry points, which resolve to
/// `num::accel` (the build's best available backend) automatically.
#pragma once

#include "container/matrix.hpp"
#include "container/matrix_ops.hpp"
#include "container/vector.hpp"
#include "container/vector_ops.hpp"
#include "core/policy.hpp"
#include "linear/factorization/lu.hpp"
#include <cmath>
#include <stdexcept>

#include <ostream>

namespace num {

/// @brief Result of a single-eigenvalue iteration
struct power_result {
    real eigenvalue = 0.0;  ///< Converged eigenvalue (Rayleigh quotient)
    vec eigenvector;     ///< Corresponding unit eigenvector
    idx iterations = 0;     ///< Iterations performed
    bool converged = false; ///< Whether tolerance was met

    friend std::ostream &operator<<(std::ostream &os, const power_result &r) {
        os << "power_result{ eigenvalue: " << r.eigenvalue
           << ", iterations: " << r.iterations
           << ", converged: " << (r.converged ? "true" : "false") << " }";
        return os;
    }
};

namespace detail {
/// Normalise v in-place; returns the old norm.
inline real normalise(vec &v) {
    // nrm <- ||v||_2
    const real nrm = kernel::norm(v.data(), v.size());
    if (nrm > 1e-300) {
        // v <- v/||v||_2
        kernel::scale(v.data(), real(1) / nrm, v.size());
    }
    return nrm;
}
} // namespace detail

/// @brief Power iteration  -- finds the eigenvalue largest in absolute value.
///
/// @param A        Square matrix (need not be symmetric)
/// @param tol      Tolerance on eigenvalue change between iterations
/// @param max_iter Maximum iterations
power_result power_iteration(const mat &A, real tol = 1e-10, idx max_iter = 1000);

/// @brief Inverse iteration  -- finds the eigenvalue closest to a shift sigma.
///
/// Factorizes (A - sigmaI) once then solves repeatedly.
///
/// @param A        Square matrix (symmetric recommended)
/// @param sigma    Shift  -- should be near the target eigenvalue
/// @param tol      Tolerance on eigenvalue change between iterations
/// @param max_iter Maximum iterations
power_result inverse_iteration(const mat &A, real sigma, real tol = 1e-10, idx max_iter = 1000);

/// @brief Rayleigh quotient iteration  -- cubically convergent.
///
/// Updates the shift sigma = v^T*A*v at every step -> fresh LU each iteration.
///
/// @param A        Symmetric matrix
/// @param x0       Starting vector (determines which eigenvalue is found)
/// @param tol      Tolerance on residual ||A*v - lambda*v||
/// @param max_iter Maximum iterations
power_result rayleigh_iteration(const mat &A, const vec &x0, real tol = 1e-10,
                               idx max_iter = 50);

inline power_result power_iteration(const mat &A, real tol, idx max_iter) {
    constexpr real tiny = 1e-300;
    const idx n = A.rows();
    if (A.cols() != n) {
        throw std::invalid_argument("power_iteration: matrix must be square");
    }

    vec v(n, 0.0);
    v[0] = 1.0;

    real lambda = 0.0;
    power_result result{0.0, v, 0, false};

    for (idx iter = 0; iter < max_iter; ++iter) {
        result.iterations = iter + 1;

        vec w(n);
        matvec(A, v, w);

        real new_lambda = dot(v, w);
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

inline power_result inverse_iteration(const mat &A, real sigma, real tol, idx max_iter) {
    constexpr real tiny = 1e-300;
    const idx n = A.rows();
    if (A.cols() != n) {
        throw std::invalid_argument("inverse_iteration: matrix must be square");
    }

    // Factorize (A - sigma*I) once
    mat M = A;
    for (idx i = 0; i < n; ++i) {
        M(i, i) -= sigma;
    }
    // M is A (rejected above unless square) shifted along its diagonal.
    lu_result f = lu(assume_square(M));

    vec v(n, 0.0);
    v[0] = 1.0;

    real lambda = 0.0;
    power_result result{0.0, v, 0, false};

    for (idx iter = 0; iter < max_iter; ++iter) {
        result.iterations = iter + 1;

        vec w(n);
        lu_solve(f, v, w);
        detail::normalise(w);

        // Rayleigh quotient as eigenvalue estimate
        vec av(n);
        matvec(A, w, av);
        real new_lambda = dot(w, av);

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

inline power_result rayleigh_iteration(const mat &A, const vec &x0, real tol, idx max_iter) {
    const idx n = A.rows();
    if (A.cols() != n) {
        throw std::invalid_argument("rayleigh_iteration: matrix must be square");
    }
    if (x0.size() != n) {
        throw std::invalid_argument("rayleigh_iteration: x0 size mismatch");
    }

    vec v = x0;
    detail::normalise(v);

    // Initial Rayleigh quotient
    vec av(n);
    matvec(A, v, av);
    real sigma = dot(v, av);

    power_result result{sigma, v, 0, false};

    for (idx iter = 0; iter < max_iter; ++iter) {
        result.iterations = iter + 1;

        // Factorize (A - sigma*I); fresh each iteration (cubic convergence)
        mat M = A;
        for (idx i = 0; i < n; ++i) {
            M(i, i) -= sigma;
        }
        // M is A (rejected above unless square) shifted along its diagonal.
        lu_result f = lu(assume_square(M));

        if (f.singular) {
            break;
        }

        vec w(n);
        lu_solve(f, v, w);
        detail::normalise(w);

        matvec(A, w, av);
        real new_sigma = dot(w, av);

        // res <- ||A*w - sigma*w||_2
        const real res = std::sqrt(
            kernel::linear_combination_norm_sq(av.data(), real(1), w.data(), -new_sigma, n));

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
