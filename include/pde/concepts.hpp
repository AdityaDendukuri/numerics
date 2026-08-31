/// @file pde/concepts.hpp
/// @brief Contracts for stencils, grid operators, and time steppers.
#pragma once

#include "algebra/concepts.hpp"
#include "algebra/properties.hpp"
#include "container/vector.hpp"
#include "core/types.hpp"
#include "operator/concepts.hpp"
#include <concepts>

namespace num {

/// @brief Finite-difference stencil applied over a structured grid.
///
/// A stencil writes \f$(Lu)_i\f$ from a neighbourhood of \f$u_i\f$ without
/// assembling a matrix, which is what keeps the work \f$O(N)\f$ per step for a
/// grid of \f$N\f$ points.
template <class S>
concept GridStencil = requires(const S &stencil, const Vector &u, Vector &out, int n) {
    stencil.apply(u, out, n);
};

/// @brief Grid operator that can also materialize itself as a sparse matrix.
///
/// Krylov methods need only the action. A direct solve needs the matrix. An
/// operator satisfying this supports both, so the choice of solver does not
/// change how the discretization is written.
template <class Op>
concept AssemblableGridOperator = LinearOperator<Op> && requires(const Op &A) {
    { A.to_sparse() };
};

/// @brief Operator arising from an implicit step, \f$(I - \Delta t\, L)\f$.
///
/// For a diffusion operator \f$L\f$ this is symmetric positive definite for every
/// \f$\Delta t > 0\f$, which is what allows conjugate gradients rather than a
/// general solver. The property is declared through the hierarchy, so a stepper
/// constrained here cannot be handed an operator that lacks it.
template <class Op>
concept ImplicitStepOperator = SPDOperator<Op>;

/// @brief Stepper advancing a field from \f$t\f$ to \f$t + \Delta t\f$.
template <class S, class Field>
concept FieldStepper = requires(S &stepper, Field &u, real dt) {
    stepper.step(u, dt);
};

} // namespace num
