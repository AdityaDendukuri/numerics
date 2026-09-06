/// @file math_pcg.hpp
/// @brief Generic PCG constrained by operator and preconditioner evidence.
#pragma once

#include "core/math/concepts.hpp"
#include "core/math/evidence.hpp"
#include "core/math/subspace.hpp"
#include "linear/solvers/solver_result.hpp"
#include <cmath>
#include <concepts>
#include <stdexcept>

namespace num {

struct pcg_options {
    real tolerance = 1e-10;
    idx max_iterations = 1000;
};

namespace math_krylov_detail {

template <class Op, class M, class V, class Invariant>
requires math::inner_product_space<V> &&math::endomorphism_on<Op, V> &&math::endomorphism_on<M, V> &&
    std::floating_point<math::scalar_t<V>> [[nodiscard]] solver_result
    pcg_recurrence(const Op &A, const M &preconditioner, const V &b, V &x, pcg_options options,
                   const Invariant &check_invariant) {
    using S = math::scalar_t<V>;
    const auto n = math::dimension(b);
    if (math::dimension(x) != n || A.rows() != n || A.cols() != n || preconditioner.rows() != n ||
        preconditioner.cols() != n) {
        throw std::invalid_argument(
            "pcg: incompatible operator, preconditioner, and vector dimensions");
    }
    if (!(options.tolerance > 0.0)) {
        throw std::invalid_argument("pcg: invalid convergence options");
    }
    V residual = math::zero_like(b);
    V preconditioned = math::zero_like(b);
    V direction = math::zero_like(b);
    V applied = math::zero_like(b);

    math::apply(A, x, residual);
    check_invariant(residual, "pcg: operator did not preserve the certified subspace");
    // r <- b - A*x
    math::linear_combination(S(1), b, S(-1), residual);
    check_invariant(residual, "pcg: residual left the certified subspace");

    solver_result result{0, static_cast<real>(math::norm(residual)), false};
    if (result.residual < options.tolerance) {
        result.converged = true;
        return result;
    }

    math::apply(preconditioner, residual, preconditioned);
    check_invariant(preconditioned, "pcg: preconditioner did not preserve the certified subspace");
    S weighted_residual = math::inner(residual, preconditioned);
    if (!(weighted_residual > S(0)) || !std::isfinite(weighted_residual)) {
        throw std::runtime_error("pcg: positive-definite preconditioner invariant was violated");
    }
    direction = preconditioned;

    for (idx iteration = 0; iteration < options.max_iterations; ++iteration) {
        result.iterations = iteration + 1;
        math::apply(A, direction, applied);
        check_invariant(applied, "pcg: operator did not preserve the certified subspace");
        const S curvature = math::inner(direction, applied);
        if (!(curvature > S(0)) || !std::isfinite(curvature)) {
            throw std::runtime_error("pcg: positive-definite operator invariant was violated");
        }

        const S alpha = weighted_residual / curvature;
        // x <- x + alpha*p
        math::axpy(alpha, direction, x);
        // r <- r - alpha*A*p; ||r||_2^2
        const S residual_square = math::axpy_norm_sq(-alpha, applied, residual);
        check_invariant(x, "pcg: iterate left the certified subspace");
        check_invariant(residual, "pcg: residual left the certified subspace");
        result.residual = static_cast<real>(std::sqrt(residual_square));
        if (!std::isfinite(result.residual)) {
            throw std::runtime_error("pcg: inner-product norm invariant was violated");
        }
        if (result.residual < options.tolerance) {
            result.converged = true;
            break;
        }

        math::apply(preconditioner, residual, preconditioned);
        check_invariant(preconditioned,
                        "pcg: preconditioner did not preserve the certified subspace");
        const S next_weighted_residual = math::inner(residual, preconditioned);
        if (!(next_weighted_residual > S(0)) || !std::isfinite(next_weighted_residual)) {
            throw std::runtime_error(
                "pcg: positive-definite preconditioner invariant was violated");
        }
        const S beta = next_weighted_residual / weighted_residual;
        // p <- M^-1*r + beta*p
        math::linear_combination(S(1), preconditioned, beta, direction);
        check_invariant(direction, "pcg: search direction left the certified subspace");
        weighted_residual = next_weighted_residual;
    }
    return result;
}

} // namespace math_krylov_detail

/// PCG on the whole vector space. Both A and the approximate inverse M must be
/// globally positive definite.
template <class Op, class M, class V>
requires math::inner_product_space<V> &&math::endomorphism_on<Op, V> &&math::endomorphism_on<M, V> &&
    claims<Op, law::spd> &&claims<M, law::spd> &&
        std::floating_point<math::scalar_t<V>> [[nodiscard]] solver_result
        pcg(const Op &A, const M &preconditioner, const V &b, V &x, pcg_options options = {}) {
    const auto no_restriction = [](const V &, const char *) {};
    return math_krylov_detail::pcg_recurrence(A, preconditioner, b, x, options, no_restriction);
}

/// PCG on a named invariant subspace. Evidence for A and M is tied to the exact
/// same Subspace type, and membership is checked throughout the recurrence.
template <class Subspace, class Op, class M, class V>
requires math::inner_product_space<V> &&math::linear_subspace_of<Subspace, V> &&
    math::endomorphism_on<Op, V> &&math::endomorphism_on<M, V> &&
        claims<Op, law::spd_on<Subspace>> &&
            claims<M, law::spd_on<Subspace>> &&
                std::floating_point<math::scalar_t<V>> [[nodiscard]] solver_result
                pcg(const Op &A, const M &preconditioner, const V &b, V &x,
                    const Subspace &subspace, pcg_options options = {}) {
    if (!math::contains(subspace, b)) {
        throw std::invalid_argument("pcg: right-hand side is outside the certified subspace");
    }
    if (!math::contains(subspace, x)) {
        throw std::invalid_argument("pcg: initial iterate is outside the certified subspace");
    }
    const auto enforce_subspace = [&subspace](const V &value, const char *message) {
        if (!math::contains(subspace, value)) {
            throw std::runtime_error(message);
        }
    };
    return math_krylov_detail::pcg_recurrence(A, preconditioner, b, x, options, enforce_subspace);
}

} // namespace num
