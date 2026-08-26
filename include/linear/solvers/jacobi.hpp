/// @file jacobi.hpp
/// @brief Jacobi iterative solver
#pragma once

#include <cmath>
#include <stdexcept>
#include "container/vector_ops.hpp"
#include "container/matrix.hpp"
#include "core/policy.hpp"
#include "container/vector.hpp"
#include "linear/solvers/solver_result.hpp"

namespace num {

/// @brief Jacobi iterative solver for Ax = b
///
/// Updates all components simultaneously using only values from the previous
/// iteration. Converges for strictly diagonally dominant A. Trivially
/// parallelisable with backend::omp.
///
/// @param A        Square matrix
/// @param b        Right-hand side vector
/// @param x        Solution vector (initial guess on input, solution on output)
/// @param tol      Convergence tolerance on residual norm (default 1e-10)
/// @param max_iter Maximum iterations (default 1000)
/// @param backend  Execution backend (default: backend::dflt)
/// @return SolverResult with convergence info
SolverResult jacobi(const Matrix &A, const Vector &b, Vector &x, real tol = 1e-10,
                    idx max_iter = 1000, Backend backend = backend::dflt);



inline SolverResult jacobi(const Matrix &A, const Vector &b, Vector &x, real tol, idx max_iter,
                    Backend backend) {
    constexpr real zero_diag_tol = 1e-15;
    idx n = b.size();
    if (A.rows() != n || A.cols() != n || x.size() != n) {
        throw std::invalid_argument("Dimension mismatch in Jacobi solver");
    }

    Vector x_new(n);
    SolverResult result{0, 0.0, false};

    for (idx iter = 0; iter < max_iter; ++iter) {
        // Compute all updates from the previous iterate simultaneously
#ifdef NUMERICS_HAS_OMP
#pragma omp parallel for schedule(static) if (backend == backend::omp)
#endif
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

        for (idx i = 0; i < n; ++i) {
            x[i] = x_new[i];
        }

        // Residual ||b - Ax||
        real res = 0.0;
#ifdef NUMERICS_HAS_OMP
#pragma omp parallel for reduction(+ : res) schedule(static) if (backend == backend::omp)
#endif
        for (idx i = 0; i < n; ++i) {
            real ri = b[i];
            for (idx j = 0; j < n; ++j) {
                ri -= A(i, j) * x[j];
            }
            res += ri * ri;
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
