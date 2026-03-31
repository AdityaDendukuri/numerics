/// @file cg.hpp
/// @brief Conjugate gradient solvers (dense and matrix-free)
#pragma once
#include "linalg/solvers/solver_result.hpp"
#include "core/matrix.hpp"
#include "core/vector.hpp"
#include "core/policy.hpp"
#include <functional>

namespace num {

/// @brief Callable type for matrix-free matvec: computes y = A*x
using MatVecFn = std::function<void(const Vector&, Vector&)>;

/// @brief Conjugate gradient solver for Ax = b
/// @param A        Symmetric positive definite matrix
/// @param b        Right-hand side vector
/// @param x        Solution vector (initial guess on input, solution on output)
/// @param tol      Convergence tolerance on residual norm (default 1e-10)
/// @param max_iter Maximum iterations (default 1000)
/// @param backend  Backend for internal matvec/dot/axpy/scale (default:
/// default_backend)
/// @return SolverResult with convergence info
SolverResult cg(const Matrix& A,
                const Vector& b,
                Vector&       x,
                real          tol      = 1e-10,
                idx           max_iter = 1000,
                Backend       backend  = default_backend);

/// @brief Matrix-free conjugate gradient for Ax = b where A is SPD
/// @param matvec  Callable computing y = A*x
/// @param b       Right-hand side
/// @param x       Initial guess (modified in-place -> solution)
/// @param tol     Convergence tolerance on residual norm
/// @param max_iter Maximum CG iterations
SolverResult cg_matfree(MatVecFn      matvec,
                        const Vector& b,
                        Vector&       x,
                        real          tol      = 1e-6,
                        idx           max_iter = 1000);

} // namespace num
