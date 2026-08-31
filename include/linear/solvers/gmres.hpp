/// @file linear/solvers/gmres.hpp
/// @brief Evidence-constrained restarted GMRES.
#pragma once

#include "core/policy.hpp"
#include "linear/math_adapters.hpp"
#include "linear/solvers/math_gmres.hpp"
#include "linear/sparse/sparse_op.hpp"
#include "operator/dense.hpp"

namespace num {

/// @brief Solve general unsymmetric linear systems \f$A x = b\f$ using restarted GMRES.
///
/// Constructs an orthogonal Arnoldi basis \f$V_m\f$ and solves the projected least-squares
/// problem \f$\min \| \beta e_1 - \bar{H}_m y \|_2\f$ with Givens plane rotations.
///
/// @tparam Op Linear operator type satisfying `math::EndomorphismOn<Op, Vector>`.
/// @param A General square linear operator (dense, sparse, or matrix-free).
/// @param b Right-hand side vector.
/// @param x Solution vector (serves as initial guess on input, updated in place).
/// @param tolerance Relative convergence tolerance on Euclidean residual norm \f$\|b - A x\|_2\f$.
/// @param max_iterations Maximum total Arnoldi iterations.
/// @param restart Number of Krylov subspace vectors before restarting Arnoldi process (default: 30).
/// @return `SolverResult` containing iteration count, final residual norm, and convergence boolean.
/// @throws std::invalid_argument If dimensions do not match.
/// @see cg, minres, pcg
template <class Op>
requires math::InnerProductSpace<Vector> &&math::EndomorphismOn<Op, Vector> inline SolverResult
gmres(const Op &A, const Vector &b, Vector &x, real tolerance, idx max_iterations = 1000,
      idx restart = 30) {
    return gmres(
        A, b, x,
        GMRESOptions{.tolerance = tolerance, .max_iterations = max_iterations, .restart = restart});
}

/// @brief Solve sparse linear systems \f$A x = b\f$ using restarted GMRES.
///
/// @param A Square CSR sparse matrix.
/// @param b Right-hand side vector.
/// @param x Solution vector (serves as initial guess on input, updated in place).
/// @param tolerance Convergence tolerance on Euclidean residual norm (default: 1e-6).
/// @param max_iterations Maximum total iterations (default: 1000).
/// @param restart Krylov subspace restart cycle (default: 30).
/// @return `SolverResult` containing iteration count, final residual norm, and convergence boolean.
/// @throws std::invalid_argument If `A` is not square or dimensions mismatch.
inline SolverResult gmres(const SparseMatrix &A, const Vector &b, Vector &x, real tolerance = 1e-6,
                          idx max_iterations = 1000, idx restart = 30) {
    if (A.n_rows() != A.n_cols()) {
        throw std::invalid_argument("gmres: sparse matrix must be square");
    }
    return gmres(
        operators::SparseOp(A), b, x,
        GMRESOptions{.tolerance = tolerance, .max_iterations = max_iterations, .restart = restart});
}

/// @brief Solve dense linear systems \f$A x = b\f$ using restarted GMRES with hardware backend dispatch.
///
/// @param A Square dense matrix.
/// @param b Right-hand side vector.
/// @param x Solution vector (serves as initial guess on input, updated in place).
/// @param tolerance Convergence tolerance on residual norm (default: 1e-6).
/// @param max_iterations Maximum total iterations (default: 1000).
/// @param restart Krylov subspace restart cycle (default: 30).
/// @param selected_backend Hardware execution backend (default: `backend::dflt`).
/// @return `SolverResult` containing iteration count, final residual norm, and convergence boolean.
/// @throws std::invalid_argument If `A` is not square or dimensions mismatch.
inline SolverResult gmres(const Matrix &A, const Vector &b, Vector &x, real tolerance = 1e-6,
                          idx max_iterations = 1000, idx restart = 30,
                          Backend selected_backend = backend::dflt) {
    if (A.rows() != A.cols()) {
        throw std::invalid_argument("gmres: dense matrix must be square");
    }
    return gmres(
        operators::DenseOp(A, selected_backend), b, x,
        GMRESOptions{.tolerance = tolerance, .max_iterations = max_iterations, .restart = restart});
}

} // namespace num
