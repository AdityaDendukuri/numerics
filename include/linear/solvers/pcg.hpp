/// @file linear/solvers/pcg.hpp
/// @brief Evidence-constrained preconditioned conjugate gradient.
#pragma once

#include "container/vector.hpp"
#include "core/policy.hpp"
#include "linear/solvers/math_pcg.hpp"

namespace num {

/// @brief Solve \f$A x = b\f$ using Preconditioned Conjugate Gradient (PCG).
///
/// Accelerates CG convergence using a symmetric positive-definite preconditioner \f$M \approx A^{-1}\f$.
/// Both the system operator `A` and the preconditioner `M` must carry positive-definite evidence.
///
/// @tparam Op System operator type satisfying `math::endomorphism_on<Op, vec>` and carrying SPD evidence.
/// @tparam M preconditioner operator type satisfying `math::endomorphism_on<M, vec>` and carrying SPD evidence.
/// @param A Symmetric positive-definite linear operator or matrix wrapper.
/// @param preconditioner preconditioner operator approximating \f$A^{-1}\f$ (e.g. Jacobi, Incomplete Cholesky).
/// @param b Right-hand side vector.
/// @param x Solution vector (serves as initial guess on input, updated in place).
/// @param tolerance Relative convergence tolerance on preconditioned residual norm \f$\|b - A x\|_M\f$.
/// @param max_iterations Maximum number of iterations before termination.
/// @return `solver_result` containing iteration count, final residual norm, and convergence boolean.
/// @throws std::invalid_argument If dimensions of `A`, `preconditioner`, `b`, and `x` do not match.
/// @see cg, minres, gmres, approx_chol_preconditioner
template <class Op, class M>
requires math::inner_product_space<vec> &&math::endomorphism_on<Op, vec> &&
    math::endomorphism_on<M, vec> &&claims<Op, law::spd> &&
        claims<M, law::spd> inline solver_result
        pcg(const Op &A, const M &preconditioner, const vec &b, vec &x, real tolerance,
            idx max_iterations = 1000) {
    return pcg(A, preconditioner, b, x,
               pcg_options{.tolerance = tolerance, .max_iterations = max_iterations});
}

} // namespace num
