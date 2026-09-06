/// @file jacobi.hpp
/// @brief Jacobi iterative solver
#pragma once

#include "container/matrix.hpp"
#include "container/vector.hpp"
#include "container/vector_ops.hpp"
#include "core/policy.hpp"
#include "linear/solvers/solver_result.hpp"
#include <cmath>
#include <stdexcept>

namespace num {

/// @brief Jacobi iterative solver for Ax = b
///
/// Updates all components simultaneously using only values from the previous
/// iteration. Converges for strictly diagonally dominant A. Trivially
/// parallelisable, unlike Gauss-Seidel.
///
/// @tparam Parallel  Thread both sweeps with OpenMP (default: whatever the
///                    build has available).
/// @param A        Square matrix
/// @param b        Right-hand side vector
/// @param x        Solution vector (initial guess on input, solution on output)
/// @param tol      Convergence tolerance on residual norm (default 1e-10)
/// @param max_iter Maximum iterations (default 1000)
/// @return solver_result with convergence info
template <bool Parallel = has_omp>
inline solver_result jacobi(const mat &A, const vec &b, vec &x, real tol = 1e-10,
                           idx max_iter = 1000) {
    constexpr real zero_diag_tol = 1e-15;
    idx n = b.size();
    if (A.rows() != n || A.cols() != n || x.size() != n) {
        throw std::invalid_argument("Dimension mismatch in Jacobi solver");
    }

    vec x_new(n);
    solver_result result{0, 0.0, false};

    for (idx iter = 0; iter < max_iter; ++iter) {
        // Compute all updates from the previous iterate simultaneously
#if defined(NUMERICS_HAS_OMP)
        if constexpr (Parallel) {
#pragma omp parallel for schedule(static)
            for (idx i = 0; i < n; ++i) {
                if (std::abs(A(i, i)) < zero_diag_tol) {
                    throw std::runtime_error("Zero diagonal in Jacobi solver at row " +
                                             std::to_string(i));
                }
                real sigma = 0.0;
                for (idx j = 0; j < n; ++j) {
                    if (j != i) {
                        sigma += A(i, j) * x[j];
                    }
                }
                x_new[i] = (b[i] - sigma) / A(i, i);
            }
        } else
#endif
        {
            for (idx i = 0; i < n; ++i) {
                if (std::abs(A(i, i)) < zero_diag_tol) {
                    throw std::runtime_error("Zero diagonal in Jacobi solver at row " +
                                             std::to_string(i));
                }
                real sigma = 0.0;
                for (idx j = 0; j < n; ++j) {
                    if (j != i) {
                        sigma += A(i, j) * x[j];
                    }
                }
                x_new[i] = (b[i] - sigma) / A(i, i);
            }
        }

        for (idx i = 0; i < n; ++i) {
            x[i] = x_new[i];
        }

        // Residual ||b - Ax||
        real res = 0.0;
#if defined(NUMERICS_HAS_OMP)
        if constexpr (Parallel) {
#pragma omp parallel for reduction(+ : res) schedule(static)
            for (idx i = 0; i < n; ++i) {
                real ri = b[i];
                for (idx j = 0; j < n; ++j) {
                    ri -= A(i, j) * x[j];
                }
                res += ri * ri;
            }
        } else
#endif
        {
            for (idx i = 0; i < n; ++i) {
                real ri = b[i];
                for (idx j = 0; j < n; ++j) {
                    ri -= A(i, j) * x[j];
                }
                res += ri * ri;
            }
        }
        result.residual = std::sqrt(res);
        result.iterations = iter + 1;

        if (result.residual < tol) {
            result.converged = true;
            break;
        }
    }
    return result;
}

} // namespace num
