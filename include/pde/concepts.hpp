/// @file pde/concepts.hpp
/// @brief Concepts for stencils, grid operators and time steppers.
#pragma once

#include "algebra/concepts.hpp"
#include "operator/properties.hpp"
#include "container/vector.hpp"
#include "core/types.hpp"
#include "ode/concepts.hpp"
#include "operator/concepts.hpp"
#include <concepts>

namespace num {

/// @brief Grid operator that can both be applied and be written as a sparse matrix.
template <class Op>
concept assemblable_grid_operator = linear_operator<Op> && requires(const Op &A) {
    { A.to_sparse() };
};

} // namespace num
