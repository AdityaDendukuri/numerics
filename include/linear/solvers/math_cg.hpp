/// @file math_cg.hpp
/// @brief Generic CG constrained by the foundational mathematical spine.
#pragma once

#include "core/math/concepts.hpp"
#include "core/math/evidence.hpp"
#include "linear/solvers/solver_result.hpp"
#include <cmath>
#include <concepts>
#include <stdexcept>

namespace num {

struct cg_options {
    real tolerance = 1e-10;
    idx max_iterations = 1000;
};

/// Conjugate gradients over any certified real inner-product space.
///
/// This is the canonical implementation: mathematical requirements are checked
/// here and representation-specific operations lower through the shared CPOs.
template <class Op, class V>
requires math::inner_product_space<V> &&math::endomorphism_on<Op, V> &&
    claims<Op, law::spd> &&
        std::floating_point<math::scalar_t<V>> [[nodiscard]] solver_result
        cg(const Op &A, const V &b, V &x, cg_options options = {}) {
    using S = math::scalar_t<V>;

    const auto n = math::dimension(b);
    if (math::dimension(x) != n || A.rows() != n || A.cols() != n) {
        throw std::invalid_argument("cg: incompatible operator and vector dimensions");
    }
    if (!(options.tolerance > 0.0)) {
        throw std::invalid_argument("cg: invalid convergence options");
    }

    V residual = math::zero_like(b);
    V direction = math::zero_like(b);
    V applied = math::zero_like(b);

    math::apply(A, x, residual);
    // r <- b - A*x
    math::linear_combination(S(1), b, S(-1), residual);
    direction = residual;

    S residual_square = math::inner(residual, residual);
    solver_result result{0, static_cast<real>(std::sqrt(residual_square)), false};
    if (result.residual < options.tolerance) {
        result.converged = true;
        return result;
    }

    for (idx iteration = 0; iteration < options.max_iterations; ++iteration) {
        result.iterations = iteration + 1;
        math::apply(A, direction, applied);

        const S curvature = math::inner(direction, applied);
        if (!(curvature > S(0)) || !std::isfinite(curvature)) {
            throw std::runtime_error("cg: positive-definite curvature invariant was violated");
        }

        const S alpha = residual_square / curvature;
        // x <- x + alpha*p
        math::axpy(alpha, direction, x);

        // r <- r - alpha*A*p; ||r||_2^2
        const S next_square = math::axpy_norm_sq(-alpha, applied, residual);
        if (!(next_square >= S(0)) || !std::isfinite(next_square)) {
            throw std::runtime_error("cg: inner-product norm invariant was violated");
        }
        result.residual = static_cast<real>(std::sqrt(next_square));
        if (result.residual < options.tolerance) {
            result.converged = true;
            break;
        }

        const S beta = next_square / residual_square;
        // p <- r + beta*p
        math::linear_combination(S(1), residual, beta, direction);
        residual_square = next_square;
    }

    return result;
}

} // namespace num

namespace num::experimental {

using ::num::cg;
using ::num::cg_options;

} // namespace num::experimental
