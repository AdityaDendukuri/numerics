/// @file linear/solvers/minres.hpp
/// @brief Evidence-constrained minimum residual iteration.
#pragma once

#include "container/vector.hpp"
#include "core/policy.hpp"
#include "linear/solvers/math_minres.hpp"

namespace num {

/// Compatibility spelling for callers passing scalar convergence parameters.
template <class Op>
requires math::InnerProductSpace<Vector> &&math::EndomorphismOn<Op, Vector> &&
    math::Carries<Op, axiom::self_adjoint> inline SolverResult
    minres(const Op &A, const Vector &b, Vector &x, real tolerance, idx max_iterations = 1000,
           Backend = backend::dflt) {
    return minres(A, b, x, MINRESOptions{.tolerance = tolerance, .max_iterations = max_iterations});
}

} // namespace num
