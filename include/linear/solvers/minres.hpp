/// @file linear/solvers/minres.hpp
/// @brief Evidence-constrained minimum residual iteration.
#pragma once

#include "container/vector.hpp"
#include "core/policy.hpp"
#include "linear/solvers/math_minres.hpp"

namespace num {

/// @brief Solve \f$A x = b\f$ using MINRES (Minimum Residual) for symmetric / self-adjoint systems.
///
/// Minimizes the 2-norm of the residual \f$\|b - A x_k\|_2\f$ over the Krylov subspace \f$\mathcal{K}_k(A, r_0)\f$.
/// Unlike CG, MINRES converges stably on symmetric **indefinite** linear systems.
///
/// @tparam Op Linear operator type satisfying `math::endomorphism_on<Op, vec>` and carrying self-adjoint evidence.
/// @param A Symmetric / self-adjoint linear operator (e.g. `num::assume_symmetric(A)`).
/// @param b Right-hand side vector.
/// @param x Solution vector (serves as initial guess on input, updated in place).
/// @param tolerance Convergence tolerance on Euclidean residual norm \f$\|b - A x\|_2\f$.
/// @param max_iterations Maximum number of Lanczos iterations before termination.
/// @return `solver_result` containing iteration count, final residual norm, and convergence boolean.
/// @throws std::invalid_argument If dimensions of `A`, `b`, and `x` do not match.
/// @see cg, gmres, pcg, assume_symmetric
template <class Op>
requires math::inner_product_space<vec> &&math::endomorphism_on<Op, vec> &&
    claims<Op, law::self_adjoint> inline solver_result
    minres(const Op &A, const vec &b, vec &x, real tolerance, idx max_iterations = 1000) {
    return minres(A, b, x, minres_options{.tolerance = tolerance, .max_iterations = max_iterations});
}

} // namespace num
