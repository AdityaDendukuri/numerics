/// @file linear/eigen/lanczos.hpp
/// @brief Lanczos iteration: \f$Q_m^T A Q_m = T_m\f$, \f$T_m\f$ tridiagonal.
// TODO: thick restarts and selective reorthogonalization.
#pragma once

#include "linear/sparse/sparse_op.hpp"
#include "operator/dense.hpp"
#include "operator/properties.hpp"

#include "container/vector_ops.hpp"

#include "linear/matrix_properties.hpp"
#include "linear/solvers/solver_result.hpp"
#include "operator/concepts.hpp"

#include "container/matrix.hpp"
#include "container/vector.hpp"
#include "core/policy.hpp"
#include "linear/eigen/jacobi_eig.hpp"
#include "linear/sparse/sparse.hpp"
#include "linear/subspace.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <ostream>

namespace num {

/// @brief Result of the Lanczos method: Ritz pairs and convergence.
struct lanczos_result {
    vec<real> ritz_values;        ///< Requested Ritz values in ascending order.
    mat<real> ritz_vectors;       ///< Ritz vectors stored as columns.
    idx steps = 0;          ///< Lanczos basis vectors generated.
    bool converged = false; ///< Whether all returned Ritz pairs met tolerance.

    friend std::ostream &operator<<(std::ostream &os, const lanczos_result &r) {
        os << "lanczos_result{ ritz_values: [" << r.ritz_values.size() << " eigenvalues]"
           << ", steps: " << r.steps << ", converged: " << (r.converged ? "true" : "false") << " }";
        return os;
    }
};

/// @brief Result of applying a matrix function by Lanczos:
/// \f$f(A) b\f$, \f$f\f$ a square root or its inverse.
struct lanczos_action_result {
    vec<real> value;
    idx steps = 0;
    real relative_change = 0.0;
    bool converged = false;
};

namespace detail {

[[nodiscard]] inline vec<real> inverse_sqrt_from_projection(const mat<real> &basis, const vec<real> &alpha,
                                                      const vec<real> &beta, idx steps,
                                                      real right_hand_side_norm) {
    mat<real> tridiagonal(steps, steps, 0.0);
    for (idx j = 0; j < steps; ++j) {
        tridiagonal(j, j) = alpha[j];
        if (j + 1 < steps) {
            tridiagonal(j, j + 1) = beta[j];
            tridiagonal(j + 1, j) = beta[j];
        }
    }

    const eigen_result eig = eig_sym(with_law<mat<real>, law::self_adjoint>(tridiagonal), 1e-13);
    vec<real> coefficients(steps, 0.0);
    for (idx eigenvector = 0; eigenvector < steps; ++eigenvector) {
        if (!(eig.values[eigenvector] > 0.0)) {
            throw std::runtime_error(
                "inverse_sqrt_lanczos: projected operator is not positive definite");
        }
        const real weight = eig.vectors(0, eigenvector) / std::sqrt(eig.values[eigenvector]);
        for (idx j = 0; j < steps; ++j) {
            coefficients[j] += eig.vectors(j, eigenvector) * weight;
        }
    }

    vec<real> value(basis.rows(), 0.0);
    for (idx row = 0; row < basis.rows(); ++row) {
        for (idx j = 0; j < steps; ++j) {
            value[row] += right_hand_side_norm * basis(row, j) * coefficients[j];
        }
    }
    return value;
}

[[nodiscard]] inline vec<real> sqrt_from_projection(const mat<real> &basis, const vec<real> &alpha, const vec<real> &beta,
                                              idx steps, real right_hand_side_norm) {
    mat<real> tridiagonal(steps, steps, 0.0);
    for (idx j = 0; j < steps; ++j) {
        tridiagonal(j, j) = alpha[j];
        if (j + 1 < steps) {
            tridiagonal(j, j + 1) = beta[j];
            tridiagonal(j + 1, j) = beta[j];
        }
    }

    const eigen_result eig = eig_sym(with_law<mat<real>, law::self_adjoint>(tridiagonal), 1e-13);
    vec<real> coefficients(steps, 0.0);
    for (idx eigenvector = 0; eigenvector < steps; ++eigenvector) {
        if (!(eig.values[eigenvector] > 0.0)) {
            throw std::runtime_error("sqrt_lanczos: projected operator is not positive definite");
        }
        const real weight = eig.vectors(0, eigenvector) * std::sqrt(eig.values[eigenvector]);
        for (idx j = 0; j < steps; ++j) {
            coefficients[j] += eig.vectors(j, eigenvector) * weight;
        }
    }

    vec<real> value(basis.rows(), 0.0);
    for (idx row = 0; row < basis.rows(); ++row) {
        for (idx j = 0; j < steps; ++j) {
            value[row] += right_hand_side_norm * basis(row, j) * coefficients[j];
        }
    }
    return value;
}

template <class Op>
requires linear_operator<Op, vec<real>, vec<real>> lanczos_action_result
inverse_sqrt_lanczos_impl(const Op &A, const vec<real> &right_hand_side, real tolerance, idx max_steps) {
    const idx n = A.rows();
    if (A.cols() != n || right_hand_side.size() != n)
        throw std::invalid_argument("inverse_sqrt_lanczos: dimension mismatch");
    if (!(tolerance > 0.0))
        throw std::invalid_argument("inverse_sqrt_lanczos: tolerance must be positive");
    if (max_steps == 0)
        max_steps = std::min<idx>(64, n);
    max_steps = std::min(max_steps, n);

    const real right_hand_side_norm = norm(right_hand_side);
    if (right_hand_side_norm == 0.0)
        return {vec<real>(n, 0.0), 0, 0.0, true};

    mat<real> basis(n, max_steps, 0.0);
    for (idx row = 0; row < n; ++row)
        basis(row, 0) = right_hand_side[row] / right_hand_side_norm;
    vec<real> alpha(max_steps, 0.0);
    vec<real> beta(max_steps, 0.0);
    vec<real> previous;

    constexpr idx minimum_steps_for_convergence_check = 4;
    for (idx j = 0; j < max_steps; ++j) {
        vec<real> direction(n);
        kernel::copy_strided(direction.data(), 1, &basis(0, j), basis.cols(), n);
        vec<real> image(n, 0.0);
        A.apply(direction, image);

        alpha[j] = dot(direction, image);
        axpy(-alpha[j], direction, image);
        if (j > 0)
            kernel::axpy_strided(image.data(), 1, &basis(0, j - 1), basis.cols(), -beta[j - 1], n);

        const real next_beta = dispatch::subspace::mgs_orthogonalize(basis, j + 1, image);
        const idx steps = j + 1;
        const bool breakdown = next_beta <= real(1e-14);
        const bool geometric_checkpoint =
            steps >= minimum_steps_for_convergence_check && (steps & (steps - 1)) == 0;
        const bool check = geometric_checkpoint || breakdown || steps == max_steps;
        real relative_change = 0.0;
        if (check) {
            const vec<real> approximation =
                inverse_sqrt_from_projection(basis, alpha, beta, steps, right_hand_side_norm);
            if (previous.size() != 0) {
                vec<real> difference = approximation;
                axpy(-1.0, previous, difference);
                relative_change = norm(difference) / std::max(norm(approximation), real(1e-30));
                if (relative_change <= tolerance) {
                    return {approximation, steps, relative_change, true};
                }
            }
            if (breakdown)
                return {approximation, steps, relative_change, true};
            if (steps == max_steps)
                return {approximation, steps, relative_change, false};
            previous = approximation;
        }

        beta[j] = next_beta;
        for (idx row = 0; row < n; ++row)
            basis(row, j + 1) = image[row] / next_beta;
    }
    return {};
}

template <class Op>
requires linear_operator<Op, vec<real>, vec<real>> lanczos_action_result
sqrt_lanczos_impl(const Op &A, const vec<real> &right_hand_side, real tolerance, idx max_steps) {
    const idx n = A.rows();
    if (A.cols() != n || right_hand_side.size() != n)
        throw std::invalid_argument("sqrt_lanczos: dimension mismatch");
    if (!(tolerance > 0.0))
        throw std::invalid_argument("sqrt_lanczos: tolerance must be positive");
    if (max_steps == 0)
        max_steps = std::min<idx>(64, n);
    max_steps = std::min(max_steps, n);

    const real right_hand_side_norm = norm(right_hand_side);
    if (right_hand_side_norm == 0.0)
        return {vec<real>(n, 0.0), 0, 0.0, true};

    mat<real> basis(n, max_steps, 0.0);
    for (idx row = 0; row < n; ++row)
        basis(row, 0) = right_hand_side[row] / right_hand_side_norm;
    vec<real> alpha(max_steps, 0.0);
    vec<real> beta(max_steps, 0.0);
    vec<real> previous;

    constexpr idx minimum_steps_for_convergence_check = 4;
    for (idx j = 0; j < max_steps; ++j) {
        vec<real> direction(n);
        kernel::copy_strided(direction.data(), 1, &basis(0, j), basis.cols(), n);
        vec<real> image(n, 0.0);
        A.apply(direction, image);

        alpha[j] = dot(direction, image);
        axpy(-alpha[j], direction, image);
        if (j > 0)
            kernel::axpy_strided(image.data(), 1, &basis(0, j - 1), basis.cols(), -beta[j - 1], n);

        const real next_beta = dispatch::subspace::mgs_orthogonalize(basis, j + 1, image);
        const idx steps = j + 1;
        const bool breakdown = next_beta <= real(1e-14);
        const bool geometric_checkpoint =
            steps >= minimum_steps_for_convergence_check && (steps & (steps - 1)) == 0;
        const bool check = geometric_checkpoint || breakdown || steps == max_steps;
        real relative_change = 0.0;
        if (check) {
            const vec<real> approximation =
                sqrt_from_projection(basis, alpha, beta, steps, right_hand_side_norm);
            if (previous.size() != 0) {
                vec<real> difference = approximation;
                axpy(-1.0, previous, difference);
                relative_change = norm(difference) / std::max(norm(approximation), real(1e-30));
                if (relative_change <= tolerance) {
                    return {approximation, steps, relative_change, true};
                }
            }
            if (breakdown)
                return {approximation, steps, relative_change, true};
            if (steps == max_steps)
                return {approximation, steps, relative_change, false};
            previous = approximation;
        }

        beta[j] = next_beta;
        for (idx row = 0; row < n; ++row)
            basis(row, j + 1) = image[row] / next_beta;
    }
    return {};
}

template <class Op>
requires linear_operator<Op, vec<real>, vec<real>>
    lanczos_result lanczos_operator_impl(const Op &A, idx k, real tol, idx max_steps) {
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

    mat<real> V(n, max_steps, 0.0);
    vec<real> alpha(max_steps, 0.0);
    vec<real> beta(max_steps, 0.0);

    // v_0 <- e_0
    V(0, 0) = 1.0;

    idx steps = 0;

    for (idx j = 0; j < max_steps; ++j) {
        vec<real> vj(n);
        // v_j <- V[:,j]
        kernel::copy_strided(vj.data(), 1, &V(0, j), V.cols(), n);

        vec<real> w(n, 0.0);
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
    mat<real> T(m, m, 0.0);
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
    eigen_result teig = eig_sym(with_law<mat<real>, law::self_adjoint>(T), tol * real(1e-2));
    const idx nret = std::min(k, m);

    mat<real> ritz_vecs(n, nret, 0.0);
    // U_Ritz <- V_m*Z_selected
    kernel::gemm(ritz_vecs.data(), ritz_vecs.cols(), V.data(), V.cols(), &teig.vectors(0, m - nret),
                 teig.vectors.cols(), real(1), real(0), n, nret, m);

    vec<real> ritz_vals(nret);
    // lambda_Ritz <- lambda(T_m)_selected
    kernel::copy(ritz_vals.data(), teig.values.data() + (m - nret), nret);

    bool all_converged = true;
    for (idx i = 0; i < nret; ++i) {
        vec<real> u(n);
        // u_i <- U_Ritz[:,i]
        kernel::copy_strided(u.data(), 1, &ritz_vecs(0, i), ritz_vecs.cols(), n);

        vec<real> Au(n, 0.0);
        A.apply(u, Au);

        const real lam = ritz_vals[i];
        // res^2 <- ||A*u_i - lambda_i*u_i||_2^2
        const real res = kernel::linear_combination_norm_sq(Au.data(), real(1), u.data(), -lam, n);
        if (std::sqrt(res) > tol) {
            all_converged = false;
            break;
        }
    }

    return {ritz_vals, ritz_vecs, steps, all_converged};
}

} // namespace detail

/// @brief Compute \f$A^{-1/2} b\f$ by the Lanczos method:
/// \f$A^{-1/2} b \approx \|b\|_2 Q_m T_m^{-1/2} e_1\f$.
/// \f$A\f$ must be SPD.
template <class Op>
requires spd_operator<Op, vec<real>> [[nodiscard]] lanczos_action_result
inverse_sqrt_lanczos(const Op &A, const vec<real> &right_hand_side, real tolerance = 1e-8,
                     idx max_steps = 0) {
    return detail::inverse_sqrt_lanczos_impl(A, right_hand_side, tolerance, max_steps);
}

/// @brief Compute \f$A^{1/2} b\f$ by the Lanczos method, \f$A\f$ SPD:
/// \f$A^{1/2} b \approx \|b\|_2 Q_m T_m^{1/2} e_1\f$.
template <class Op>
requires spd_operator<Op, vec<real>> [[nodiscard]] lanczos_action_result
sqrt_lanczos(const Op &A, const vec<real> &right_hand_side, real tolerance = 1e-8, idx max_steps = 0) {
    return detail::sqrt_lanczos_impl(A, right_hand_side, tolerance, max_steps);
}

/// @brief Compute \f$k\f$ extreme eigenvalues and eigenvectors by the Lanczos method:
/// \f$A v \approx \lambda v\f$.
/// @throws std::invalid_argument If `k` is out of range or `A` is not square.
template <class Op>
requires self_adjoint_operator<Op, vec<real>>
    lanczos_result lanczos(const Op &A, idx k, real tol = 1e-10, idx max_steps = 0) {
    return detail::lanczos_operator_impl(A, k, tol, max_steps);
}

namespace unsafe {

/// @brief Compute extreme eigenpairs of a dense matrix by Lanczos, without checking symmetry:
/// \f$A v \approx \lambda v\f$.
/// @warning On an asymmetric matrix the basis loses orthogonality, and the Ritz values
/// are not eigenvalue estimates.
lanczos_result lanczos(const mat<real> &A, idx k, real tol = 1e-10, idx max_steps = 0);

/// @brief Compute extreme eigenpairs of a sparse matrix by Lanczos, without checking symmetry:
/// \f$A v \approx \lambda v\f$.
/// @warning On an asymmetric matrix the basis loses orthogonality, and the Ritz values
/// are not eigenvalue estimates.
lanczos_result lanczos(const spmat &A, idx k, real tol = 1e-10, idx max_steps = 0);

} // namespace unsafe

/// @brief Compute extreme eigenpairs of a matrix tagged symmetric, by Lanczos:
/// \f$A v \approx \lambda v\f$.
inline lanczos_result lanczos(const with_law<mat<real>, law::self_adjoint> &A, idx k, real tol = 1e-10,
                              idx max_steps = 0) {
    return unsafe::lanczos(A.base(), k, tol, max_steps);
}

/// @brief Reject an untagged matrix at compile time; tag it symmetric first.
template <class M>
requires matrix_space<M> && (!claims<M, law::self_adjoint>)
lanczos_result lanczos(const M & /*untagged*/, idx, real = 1e-10, idx = 0) {
    static_assert(claims<M, law::self_adjoint>,
                  "lanczos() requires a matrix carrying the symmetry invariant: the three-term "
                  "recurrence is a consequence of A = A^T and produces meaningless Ritz values "
                  "without it. "
                  "Establish it with num::assume_symmetric(A) or num::make_symmetric(A). "
                  "For a non-symmetric matrix use num::power_iteration(A) for the dominant pair. "
                  "To bypass deliberately, call num::unsafe::lanczos(A, k).");
    return {};
}

namespace unsafe {

inline lanczos_result lanczos(const mat<real> &A, idx k, real tol, idx max_steps) {
    if (A.rows() != A.cols()) {
        throw std::invalid_argument("lanczos: matrix must be square");
    }
    operators::dense_op op(A);
    return num::lanczos(num::assume_symmetric(op), k, tol, max_steps);
}

inline lanczos_result lanczos(const spmat &A, idx k, real tol, idx max_steps) {
    if (A.n_rows() != A.n_cols()) {
        throw std::invalid_argument("lanczos: matrix must be square");
    }
    operators::sparse_op op(A);
    return num::lanczos(num::assume_symmetric(op), k, tol, max_steps);
}

} // namespace unsafe

} // namespace num
