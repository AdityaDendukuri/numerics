/// @file krylov.hpp
/// @brief Restarted GMRES  -- a Krylov subspace solver for general Ax = b
#pragma once
#include "core/matrix.hpp"
#include "core/policy.hpp"
#include "core/vector.hpp"
#include "linalg/solvers/cg.hpp" // MatVecFn
#include "linalg/solvers/solver_result.hpp"
#include "linalg/sparse/sparse.hpp"

namespace num {

/// @brief Restarted GMRES(restart)  -- matrix-free interface
///
/// Works for any invertible A (symmetric or non-symmetric, indefinite).
/// The Krylov subspace is restarted every @p restart steps to bound memory.
///
/// @param matvec   Callable computing y = A*x
/// @param n        System dimension
/// @param b        Right-hand side
/// @param x        Initial guess (modified in-place -> solution)
/// @param tol      Convergence tolerance on residual norm
/// @param max_iter Maximum total matrix-vector products
/// @param restart  Krylov subspace size before restart (default 30)
/// @return SolverResult with convergence info
SolverResult gmres(MatVecFn matvec, idx n, const Vector &b, Vector &x,
                   real tol = 1e-6, idx max_iter = 1000, idx restart = 30);

/// @brief Restarted GMRES with a sparse (CSR) matrix
SolverResult gmres(const SparseMatrix &A, const Vector &b, Vector &x,
                   real tol = 1e-6, idx max_iter = 1000, idx restart = 30);

/// @brief Restarted GMRES with a dense matrix
/// @param backend  Backend for the internal matvec at each Arnoldi step
SolverResult gmres(const Matrix &A, const Vector &b, Vector &x, real tol = 1e-6,
                   idx max_iter = 1000, idx restart = 30,
                   Backend backend = default_backend);

} // namespace num
