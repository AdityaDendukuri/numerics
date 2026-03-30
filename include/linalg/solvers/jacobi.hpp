/// @file jacobi.hpp
/// @brief Jacobi iterative solver
#pragma once
#include "linalg/solvers/solver_result.hpp"
#include "core/matrix.hpp"
#include "core/vector.hpp"
#include "core/policy.hpp"

namespace num {

/// @brief Jacobi iterative solver for Ax = b
///
/// Updates all components simultaneously using only values from the previous
/// iteration. Converges for strictly diagonally dominant A. Trivially
/// parallelisable with Backend::omp.
///
/// @param A        Square matrix
/// @param b        Right-hand side vector
/// @param x        Solution vector (initial guess on input, solution on output)
/// @param tol      Convergence tolerance on residual norm (default 1e-10)
/// @param max_iter Maximum iterations (default 1000)
/// @param backend  Execution backend (default: default_backend)
/// @return SolverResult with convergence info
SolverResult jacobi(const Matrix& A, const Vector& b, Vector& x,
                    real tol = 1e-10, idx max_iter = 1000,
                    Backend backend = default_backend);

} // namespace num
