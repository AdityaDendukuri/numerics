/// @file pde/concepts.hpp
/// @brief Contracts for stencils, grid operators, and time steppers.
#pragma once

#include "algebra/concepts.hpp"
#include "operator/properties.hpp"
#include "container/vector.hpp"
#include "core/types.hpp"
#include "ode/concepts.hpp"
#include "operator/concepts.hpp"
#include <concepts>

namespace num {

/// @brief Grid operator that can also materialize itself as a sparse matrix.
///
/// Krylov methods need only the action. A direct solve needs the matrix. An
/// operator satisfying this supports both, so the choice of solver does not
/// change how the discretization is written.
template <class Op>
concept assemblable_grid_operator = linear_operator<Op> && requires(const Op &A) {
    { A.to_sparse() };
};

/// @brief Operator arising from an implicit step, \f$(I - \Delta t\, L)\f$.
///

} // namespace num
