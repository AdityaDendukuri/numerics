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
/// @tparam Op System operator type satisfying `math::EndomorphismOn<Op, Vector>` and carrying SPD evidence.
/// @tparam M Preconditioner operator type satisfying `math::EndomorphismOn<M, Vector>` and carrying SPD evidence.
/// @param A Symmetric positive-definite linear operator or matrix wrapper.
/// @param preconditioner Preconditioner operator approximating \f$A^{-1}\f$ (e.g. Jacobi, Incomplete Cholesky).
/// @param b Right-hand side vector.
/// @param x Solution vector (serves as initial guess on input, updated in place).
/// @param tolerance Relative convergence tolerance on preconditioned residual norm \f$\|b - A x\|_M\f$.
/// @param max_iterations Maximum number of iterations before termination.
/// @param backend Hardware execution backend tag (default: `backend::dflt`).
/// @return `SolverResult` containing iteration count, final residual norm, and convergence boolean.
/// @throws std::invalid_argument If dimensions of `A`, `preconditioner`, `b`, and `x` do not match.
/// @see cg, minres, gmres, ApproxCholPreconditioner
template <class Op, class M>
requires math::InnerProductSpace<Vector> &&math::EndomorphismOn<Op, Vector> &&
    math::EndomorphismOn<M, Vector> &&math::Carries<Op, axiom::positive_definite> &&
        math::Carries<M, axiom::positive_definite> inline SolverResult
        pcg(const Op &A, const M &preconditioner, const Vector &b, Vector &x, real tolerance,
            idx max_iterations = 1000, Backend = backend::dflt) {
    return pcg(A, preconditioner, b, x,
               PCGOptions{.tolerance = tolerance, .max_iterations = max_iterations});
}

} // namespace num
