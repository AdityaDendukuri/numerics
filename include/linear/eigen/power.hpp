/// @file linear/eigen/power.hpp
/// @brief Power, inverse and Rayleigh quotient iteration.
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

/// @brief Result of computing one eigenpair: \f$(\lambda, v)\f$ and convergence.
struct power_result {
    real eigenvalue = 0.0;  ///< Converged eigenvalue (Rayleigh quotient)
    vec<real> eigenvector;     ///< Corresponding unit eigenvector
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
/// @brief Normalize a vector in place: \f$v \leftarrow v / \|v\|_2\f$; returns \f$\|v\|_2\f$.
inline real normalise(vec<real> &v) {
    // nrm <- ||v||_2
    const real nrm = kernel::norm(v.data(), v.size());
    if (nrm > 1e-300) {
        // v <- v/||v||_2
        kernel::scale(v.data(), real(1) / nrm, v.size());
    }
    return nrm;
}
} // namespace detail

/// @brief Compute the eigenvalue of largest magnitude by power iteration: \f$\max |\lambda|\f$.
power_result power_iteration(const mat<real> &A, real tol = 1e-10, idx max_iter = 1000);

/// @brief Compute the eigenvalue nearest \f$\sigma\f$ by inverse iteration with \f$(A - \sigma
/// I)^{-1}\f$.
power_result inverse_iteration(const mat<real> &A, real sigma, real tol = 1e-10, idx max_iter = 1000);

/// @brief Compute an eigenpair by Rayleigh quotient iteration, with shift \f$\sigma_k = v_k^T A
/// v_k\f$.
/// @note Refactors \f$A - \sigma_k I\f$ at every step.
power_result rayleigh_iteration(const mat<real> &A, const vec<real> &x0, real tol = 1e-10,
                               idx max_iter = 50);

inline power_result power_iteration(const mat<real> &A, real tol, idx max_iter) {
    constexpr real tiny = 1e-300;
    const idx n = A.rows();
    if (A.cols() != n) {
        throw std::invalid_argument("power_iteration: matrix must be square");
    }

    vec<real> v(n, 0.0);
    v[0] = 1.0;

    real lambda = 0.0;
    power_result result{0.0, v, 0, false};

    for (idx iter = 0; iter < max_iter; ++iter) {
        result.iterations = iter + 1;

        vec<real> w(n);
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

inline power_result inverse_iteration(const mat<real> &A, real sigma, real tol, idx max_iter) {
    constexpr real tiny = 1e-300;
    const idx n = A.rows();
    if (A.cols() != n) {
        throw std::invalid_argument("inverse_iteration: matrix must be square");
    }

    // Factorize (A - sigma*I) once
    mat<real> M = A;
    for (idx i = 0; i < n; ++i) {
        M(i, i) -= sigma;
    }
    // M is A (rejected above unless square) shifted along its diagonal.
    lu_result<real> f = lu(M);

    vec<real> v(n, 0.0);
    v[0] = 1.0;

    real lambda = 0.0;
    power_result result{0.0, v, 0, false};

    for (idx iter = 0; iter < max_iter; ++iter) {
        result.iterations = iter + 1;

        vec<real> w(n);
        solve(f, v, w);
        detail::normalise(w);

        // Rayleigh quotient as eigenvalue estimate
        vec<real> av(n);
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

inline power_result rayleigh_iteration(const mat<real> &A, const vec<real> &x0, real tol, idx max_iter) {
    const idx n = A.rows();
    if (A.cols() != n) {
        throw std::invalid_argument("rayleigh_iteration: matrix must be square");
    }
    if (x0.size() != n) {
        throw std::invalid_argument("rayleigh_iteration: x0 size mismatch");
    }

    vec<real> v = x0;
    detail::normalise(v);

    // Initial Rayleigh quotient
    vec<real> av(n);
    matvec(A, v, av);
    real sigma = dot(v, av);

    power_result result{sigma, v, 0, false};

    for (idx iter = 0; iter < max_iter; ++iter) {
        result.iterations = iter + 1;

        // Factorize (A - sigma*I); fresh each iteration (cubic convergence)
        mat<real> M = A;
        for (idx i = 0; i < n; ++i) {
            M(i, i) -= sigma;
        }
        // M is A (rejected above unless square) shifted along its diagonal.
        lu_result<real> f = lu(M);

        if (f.singular) {
            break;
        }

        vec<real> w(n);
        solve(f, v, w);
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
