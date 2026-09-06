/// @file linear/eigen/lanczos.hpp
/// @brief Lanczos eigensolver for symmetric operators.
///
/// Builds an orthonormal basis \f$Q_m\f$ such that
/// \f$Q_m^T A Q_m = T_m\f$, with \f$T_m\f$ tridiagonal.
/// @todo Add thick-restart Lanczos and selective reorthogonalization controls.
#pragma once

#include "linear/sparse/sparse_op.hpp"
#include "operator/dense.hpp"
#include "operator/properties.hpp"

#include "container/vector_ops.hpp"

#include "linear/concepts.hpp"

#include "container/matrix.hpp"
#include "container/vector.hpp"
#include "core/policy.hpp"
#include "linear/eigen/jacobi_eig.hpp"
#include "linear/sparse/sparse.hpp"
#include "linear/subspace.hpp"
#include "operator/concepts.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <ostream>

namespace num {

/// Largest Ritz pairs and residual-based convergence metadata.
struct lanczos_result {
    vec ritz_values;     ///< Requested Ritz values in ascending order.
    mat ritz_vectors;    ///< Ritz vectors stored as columns.
    idx steps = 0;          ///< Lanczos basis vectors generated.
    bool converged = false; ///< Whether all returned Ritz pairs met tolerance.

    friend std::ostream &operator<<(std::ostream &os, const lanczos_result &r) {
        os << "lanczos_result{ ritz_values: [" << r.ritz_values.size() << " eigenvalues]"
           << ", steps: " << r.steps
           << ", converged: " << (r.converged ? "true" : "false") << " }";
        return os;
    }
};

namespace detail {

template <class Op>
requires linear_operator<Op, vec, vec> lanczos_result
lanczos_operator_impl(const Op &A, idx k, real tol, idx max_steps) {
    const idx n = A.rows();
    if (A.cols() != n) {
        throw std::invalid_argument("lanczos: operator must be square");
    }
    if (k == 0 || k > n) {
        throw std::invalid_argument("lanczos: k must satisfy 0 < k <= n");
    }

    if (max_steps == 0) {
        max_steps = std::min(3 * k, n);
    }
    max_steps = std::min(max_steps, n);

    mat V(n, max_steps, 0.0);
    vec alpha(max_steps, 0.0);
    vec beta(max_steps, 0.0);

    // v_0 <- e_0
    V(0, 0) = 1.0;

    idx steps = 0;

    for (idx j = 0; j < max_steps; ++j) {
        vec vj(n);
        // v_j <- V[:,j]
        kernel::copy_strided(vj.data(), 1, &V(0, j), V.cols(), n);

        vec w(n, 0.0);
        A.apply(vj, w);

        const real a = dot(vj, w);
        alpha[j] = a;

        axpy(-a, vj, w);
        if (j > 0) {
            // w <- w - beta_{j-1}*v_{j-1}
            kernel::axpy_strided(w.data(), 1, &V(0, j - 1), V.cols(), -beta[j - 1], n);
        }

        const real b = dispatch::subspace::mgs_orthogonalize(V, j + 1, w);
        ++steps;

        if (b < real(1e-12)) {
            break;
        }

        beta[j] = b;

        if (j + 1 < max_steps) {
            // V[:,j+1] <- w/beta_j
            kernel::scale_copy_strided(&V(0, j + 1), V.cols(), w.data(), 1, real(1) / b, n);
        }
    }

    const idx m = steps;
    mat T(m, m, 0.0);
    for (idx j = 0; j < m; ++j) {
        T(j, j) = alpha[j];
        if (j + 1 < m) {
            T(j, j + 1) = beta[j];
            T(j + 1, j) = beta[j];
        }
    }

    // T is the Lanczos tridiagonal, symmetric by construction: it is filled from a
    // single alpha/beta recurrence with T(j,j+1) and T(j+1,j) written from the same
    // beta. The invariant is established here rather than assumed downstream.
    eigen_result teig = eig_sym(linear::sym_mat<mat>(T), tol * real(1e-2));
    const idx nret = std::min(k, m);

    mat ritz_vecs(n, nret, 0.0);
    // U_Ritz <- V_m*Z_selected
    kernel::gemm(ritz_vecs.data(), ritz_vecs.cols(), V.data(), V.cols(),
                      &teig.vectors(0, m - nret), teig.vectors.cols(), real(1), real(0), n, nret,
                      m);

    vec ritz_vals(nret);
    // lambda_Ritz <- lambda(T_m)_selected
    kernel::copy(ritz_vals.data(), teig.values.data() + (m - nret), nret);

    bool all_converged = true;
    for (idx i = 0; i < nret; ++i) {
        vec u(n);
        // u_i <- U_Ritz[:,i]
        kernel::copy_strided(u.data(), 1, &ritz_vecs(0, i), ritz_vecs.cols(), n);

        vec Au(n, 0.0);
        A.apply(u, Au);

        const real lam = ritz_vals[i];
        // res^2 <- ||A*u_i - lambda_i*u_i||_2^2
        const real res =
            kernel::linear_combination_norm_sq(Au.data(), real(1), u.data(), -lam, n);
        if (std::sqrt(res) > tol) {
            all_converged = false;
            break;
        }
    }

    return {ritz_vals, ritz_vecs, steps, all_converged};
}

} // namespace detail

/// @brief Compute the largest \f$k\f$ extremal Ritz pairs of a symmetric / self-adjoint operator using Lanczos iteration.
///
/// Builds an orthonormal Krylov basis with modified Gram-Schmidt reorthogonalization, generates
/// a symmetric tridiagonal projection \f$T_m\f$, and extracts Ritz values and Ritz vectors.
///
/// @tparam Op Linear operator type satisfying `self_adjoint_operator<Op, vec, vec>`.
/// @param A Self-adjoint linear operator (matrix-free callable, sparse, or dense wrapper).
/// @param k Number of extremal eigenpairs to compute (\f$0 < k \le n\f$).
/// @param tol Residual tolerance \f$\|A v - \lambda v\|_2\f$ for declaring convergence (default: 1e-10).
/// @param max_steps Maximum Lanczos steps to perform (defaults to \f$\min(3k, n)\f$).
/// @return `lanczos_result` containing Ritz values, column Ritz vectors, steps performed, and convergence flag.
/// @throws std::invalid_argument If `k` is invalid or operator is not square.
/// @see eig_sym, power_iteration
template <class Op>
requires self_adjoint_operator<Op, vec, vec> lanczos_result
lanczos(const Op &A, idx k, real tol = 1e-10, idx max_steps = 0) {
    return detail::lanczos_operator_impl(A, k, tol, max_steps);
}

namespace unsafe {

/// @brief Lanczos on a stored dense matrix without requiring the symmetry invariant.
///
/// The three-term recurrence is derived from \f$A = A^T\f$; without it the basis
/// loses orthogonality and the Ritz values are not eigenvalue estimates.
lanczos_result lanczos(const mat &A, idx k, real tol = 1e-10, idx max_steps = 0);

/// @brief Lanczos on a stored sparse matrix without requiring the symmetry invariant.
lanczos_result lanczos(const spmat &A, idx k, real tol = 1e-10, idx max_steps = 0);

} // namespace unsafe

/// @brief Compute largest \f$k\f$ Ritz pairs of a matrix carrying certified symmetry evidence.
///
/// @param A Symmetric matrix carrying symmetry evidence (e.g. `num::assume_symmetric(A)`).
/// @param k Number of extremal eigenpairs to compute.
/// @param tol Residual tolerance (default: 1e-10).
/// @param max_steps Maximum Lanczos steps (default: \f$\min(3k, n)\f$).
/// @return `lanczos_result` with Ritz pairs and convergence metadata.
inline lanczos_result lanczos(const linear::sym_mat<mat> &A, idx k, real tol = 1e-10,
                             idx max_steps = 0) {
    return unsafe::lanczos(A.base(), k, tol, max_steps);
}

/// @brief Rejects an untagged matrix at compile time.
template <class M>
    requires matrix_space<M> &&
    (!symmetric_matrix_like<M>)lanczos_result lanczos(const M & /*untagged*/, idx, real = 1e-10,
                                                   idx = 0) {
    static_assert(symmetric_matrix_like<M>,
                  "lanczos() requires a matrix carrying the symmetry invariant: the three-term "
                  "recurrence is a consequence of A = A^T and produces meaningless Ritz values "
                  "without it. "
                  "Establish it with num::assume_symmetric(A) or num::make_symmetric(A). "
                  "For a non-symmetric matrix use num::power_iteration(A) for the dominant pair. "
                  "To bypass deliberately, call num::unsafe::lanczos(A, k).");
    return {};
}

namespace unsafe {

inline lanczos_result lanczos(const mat &A, idx k, real tol, idx max_steps) {
    if (A.rows() != A.cols()) {
        throw std::invalid_argument("lanczos: matrix must be square");
    }
    operators::dense_op op(A);
    return num::lanczos(operators::assume_symmetric(op), k, tol, max_steps);
}

inline lanczos_result lanczos(const spmat &A, idx k, real tol, idx max_steps) {
    if (A.n_rows() != A.n_cols()) {
        throw std::invalid_argument("lanczos: matrix must be square");
    }
    operators::sparse_op op(A);
    return num::lanczos(operators::assume_symmetric(op), k, tol, max_steps);
}

} // namespace unsafe

} // namespace num
