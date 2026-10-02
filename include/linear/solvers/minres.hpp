/// @file linear/solvers/minres.hpp
/// @brief Evidence-constrained minimum residual iteration.
#pragma once

#include "container/matrix.hpp"
#include "container/vector.hpp"
#include "core/math/concepts.hpp"
#include "core/policy.hpp"
#include "core/types.hpp"
#include "linear/factorization/qr.hpp"
#include "linear/solvers/solver_result.hpp"
#include <algorithm>
#include <cmath>
#include <concepts>
#include <stdexcept>
#include <utility>
#include <vector>

namespace num {

/// @brief Convergence options of `num::minres`: tolerance and iteration limit.
struct minres_options {
    real tolerance = 1e-10;
    idx max_iterations = 1000;
};

namespace math_krylov_detail {

inline vec<real> minres_projected_solve(const array<real> &alpha, const array<real> &beta,
                                     real beta0, idx m) {
    mat<real> H(m + 1, m, 0.0);
    for (idx j = 0; j < m; ++j) {
        H(j, j) = alpha[j];
        if (j > 0)
            H(j - 1, j) = beta[j - 1];
        H(j + 1, j) = beta[j];
    }
    vec<real> rhs(m + 1, 0.0);
    rhs[0] = beta0;
    const qr_result factor = qr(H);
    vec<real> y(m, 0.0);
    qr_solve(factor, rhs, y);
    return y;
}

} // namespace math_krylov_detail

/// @brief Solve \f$A x = b\f$ using MINRES (Minimum Residual) for symmetric / self-adjoint systems.
///
/// Minimizes the 2-norm of the residual \f$\|b - A x_k\|_2\f$ over the Krylov subspace \f$\mathcal{K}_k(A, r_0)\f$.
/// Unlike CG, MINRES converges stably on symmetric **indefinite** linear systems.
///
/// @param A Symmetric / self-adjoint linear operator (e.g. `num::assume_symmetric(A)`).
/// @param b Right-hand side vector.
/// @param x Solution vector (serves as initial guess on input, updated in place).
/// @param options Tolerance on \f$\|b - A x\|_2\f$ and iteration limit.
/// @return `solver_result` containing iteration count, final residual norm, and convergence boolean.
/// @throws std::invalid_argument If dimensions of `A`, `b`, and `x` do not match.
/// @see cg, gmres, pcg, assume_symmetric
template <class Op, class V>
requires math::inner_product_space<V> &&math::self_adjoint_operator<Op, V> &&
        std::same_as<math::scalar_t<V>, real> [[nodiscard]] solver_result
        minres(const Op &A, const V &b, V &x, minres_options options = {}) {
    const auto n = math::dimension(b);
    if (math::dimension(x) != n || A.rows() != n || A.cols() != n) {
        throw std::invalid_argument("minres: incompatible operator and vector dimensions");
    }
    if (!(options.tolerance > 0.0)) {
        throw std::invalid_argument("minres: invalid convergence options");
    }

    V residual = math::zero_like(b);
    V applied = math::zero_like(b);
    math::apply(A, x, residual);
    // r <- b - A*x
    math::linear_combination(real(1), b, real(-1), residual);

    const real beta0 = math::norm(residual);
    solver_result result{0, beta0, beta0 < options.tolerance};
    if (result.converged || options.max_iterations == 0)
        return result;

    const idx mmax = std::min<idx>(options.max_iterations, static_cast<idx>(n));
    array<V> basis;
    basis.reserve(mmax + 1);
    basis.push_back(residual);
    math::scale(real(1) / beta0, basis[0]);

    array<real> alpha;
    array<real> beta;
    alpha.reserve(mmax);
    beta.reserve(mmax);
    V previous = math::zero_like(b);

    for (idx j = 0; j < mmax; ++j) {
        result.iterations = j + 1;
        math::apply(A, basis[j], applied);
        if (j > 0)
            math::axpy(-beta[j - 1], previous, applied);

        const real diagonal = math::inner(basis[j], applied);
        if (!std::isfinite(diagonal)) {
            throw std::runtime_error("minres: self-adjoint Lanczos invariant was violated");
        }
        alpha.push_back(diagonal);
        math::axpy(-diagonal, basis[j], applied);

        const real next_beta = math::norm(applied);
        if (!std::isfinite(next_beta)) {
            throw std::runtime_error("minres: inner-product norm invariant was violated");
        }
        beta.push_back(next_beta);

        const vec<real> y = math_krylov_detail::minres_projected_solve(alpha, beta, beta0, j + 1);
        V candidate = x;
        for (idx column = 0; column <= j; ++column)
            math::axpy(y[column], basis[column], candidate);

        math::apply(A, candidate, residual);
        // r <- b - A*x_candidate
        math::linear_combination(real(1), b, real(-1), residual);
        result.residual = math::norm(residual);
        if (result.residual < options.tolerance) {
            x = std::move(candidate);
            result.converged = true;
            break;
        }
        if (next_beta <= real(1e-15)) {
            x = std::move(candidate);
            break;
        }

        previous = basis[j];
        math::scale(real(1) / next_beta, applied);
        basis.push_back(applied);
        if (j + 1 == mmax)
            x = std::move(candidate);
    }
    return result;
}

} // namespace num
