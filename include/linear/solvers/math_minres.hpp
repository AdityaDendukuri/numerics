/// @file math_minres.hpp
/// @brief Generic MINRES constrained by self-adjoint evidence.
#pragma once

#include "core/types.hpp"
#include "container/matrix.hpp"
#include "container/vector.hpp"
#include "core/math/concepts.hpp"
#include "core/math/evidence.hpp"
#include "linear/factorization/qr.hpp"
#include "linear/solvers/solver_result.hpp"
#include <algorithm>
#include <cmath>
#include <concepts>
#include <stdexcept>
#include <utility>
#include <vector>

namespace num {

struct minres_options {
    real tolerance = 1e-10;
    idx max_iterations = 1000;
};

namespace math_krylov_detail {

inline vec minres_projected_solve(const array<real> &alpha, const array<real> &beta,
                                     real beta0, idx m) {
    mat H(m + 1, m, 0.0);
    for (idx j = 0; j < m; ++j) {
        H(j, j) = alpha[j];
        if (j > 0)
            H(j - 1, j) = beta[j - 1];
        H(j + 1, j) = beta[j];
    }
    vec rhs(m + 1, 0.0);
    rhs[0] = beta0;
    const qr_result factor = qr(H);
    vec y(m, 0.0);
    qr_solve(factor, rhs, y);
    return y;
}

} // namespace math_krylov_detail

/// Minimum residual projection for a certified self-adjoint endomorphism.
template <class Op, class V>
requires math::inner_product_space<V> &&math::endomorphism_on<Op, V> &&
    claims<Op, law::self_adjoint> &&
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

        const vec y = math_krylov_detail::minres_projected_solve(alpha, beta, beta0, j + 1);
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
