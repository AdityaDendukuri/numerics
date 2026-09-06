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
/// @tparam Op Linear operator type satisfying `math::endomorphism_on<Op, vec>`.
/// @param A General square linear operator (dense, sparse, or matrix-free).
/// @param b Right-hand side vector.
/// @param x Solution vector (serves as initial guess on input, updated in place).
/// @param tolerance Relative convergence tolerance on Euclidean residual norm \f$\|b - A x\|_2\f$.
/// @param max_iterations Maximum total Arnoldi iterations.
/// @param restart Number of Krylov subspace vectors before restarting Arnoldi process (default: 30).
/// @return `solver_result` containing iteration count, final residual norm, and convergence boolean.
/// @throws std::invalid_argument If dimensions do not match.
/// @see cg, minres, pcg
template <class Op>
requires math::inner_product_space<vec> &&math::endomorphism_on<Op, vec> inline solver_result
gmres(const Op &A, const vec &b, vec &x, real tolerance, idx max_iterations = 1000,
      idx restart = 30) {
    return gmres(
        A, b, x,
        gmres_options{.tolerance = tolerance, .max_iterations = max_iterations, .restart = restart});
}

/// @brief Solve sparse linear systems \f$A x = b\f$ using restarted GMRES.
///
/// @param A Square CSR sparse matrix.
/// @param b Right-hand side vector.
/// @param x Solution vector (serves as initial guess on input, updated in place).
/// @param tolerance Convergence tolerance on Euclidean residual norm (default: 1e-6).
/// @param max_iterations Maximum total iterations (default: 1000).
/// @param restart Krylov subspace restart cycle (default: 30).
/// @return `solver_result` containing iteration count, final residual norm, and convergence boolean.
/// @throws std::invalid_argument If `A` is not square or dimensions mismatch.
inline solver_result gmres(const spmat &A, const vec &b, vec &x, real tolerance = 1e-6,
                          idx max_iterations = 1000, idx restart = 30) {
    if (A.n_rows() != A.n_cols()) {
        throw std::invalid_argument("gmres: sparse matrix must be square");
    }
    return gmres(
        operators::sparse_op(A), b, x,
        gmres_options{.tolerance = tolerance, .max_iterations = max_iterations, .restart = restart});
}

/// @brief Solve dense linear systems \f$A x = b\f$ using restarted GMRES with hardware backend dispatch.
///
/// @param A Square dense matrix.
/// @param b Right-hand side vector.
/// @param x Solution vector (serves as initial guess on input, updated in place).
/// @param tolerance Convergence tolerance on residual norm (default: 1e-6).
/// @param max_iterations Maximum total iterations (default: 1000).
/// @param restart Krylov subspace restart cycle (default: 30).
/// @return `solver_result` containing iteration count, final residual norm, and convergence boolean.
/// @throws std::invalid_argument If `A` is not square or dimensions mismatch.
inline solver_result gmres(const mat &A, const vec &b, vec &x, real tolerance = 1e-6,
                          idx max_iterations = 1000, idx restart = 30) {
    if (A.rows() != A.cols()) {
        throw std::invalid_argument("gmres: dense matrix must be square");
    }
    return gmres(
        operators::dense_op(A), b, x,
        gmres_options{.tolerance = tolerance, .max_iterations = max_iterations, .restart = restart});
}

} // namespace num
