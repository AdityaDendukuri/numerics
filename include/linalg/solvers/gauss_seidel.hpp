/// @file gauss_seidel.hpp
/// @brief Gauss-Seidel iterative solver
#pragma once
#include "linalg/solvers/solver_result.hpp"
#include "core/matrix.hpp"
#include "core/vector.hpp"
#include "core/policy.hpp"

namespace num {

/// @brief Gauss-Seidel iterative solver for Ax = b
///
/// Updates each component x[i] in-place using the latest values of all
/// other components. Converges for strictly diagonally dominant or symmetric
/// positive definite A.
///
/// With Backend::omp the residual computation is parallelised; the update
/// sweep remains sequential to preserve convergence properties.
///
/// @param A        Square matrix
/// @param b        Right-hand side vector
/// @param x        Solution vector (initial guess on input, solution on output)
/// @param tol      Convergence tolerance on residual norm (default 1e-10)
/// @param max_iter Maximum iterations (default 1000)
/// @param backend  Execution backend (default: default_backend)
/// @return SolverResult with convergence info
SolverResult gauss_seidel(const Matrix& A,
                          const Vector& b,
                          Vector&       x,
                          real          tol      = 1e-10,
                          idx           max_iter = 1000,
                          Backend       backend  = default_backend);

} // namespace num
