/// @file linear/solvers/pcg.hpp
/// @brief Evidence-constrained preconditioned conjugate gradient.
#pragma once

#include "container/vector.hpp"
#include "core/policy.hpp"
#include "linear/solvers/math_pcg.hpp"

namespace num {

/// Compatibility spelling for callers passing scalar convergence parameters.
/// Backend selection belongs to the operator/preconditioner adapters and is not
/// part of the mathematical recurrence.
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
