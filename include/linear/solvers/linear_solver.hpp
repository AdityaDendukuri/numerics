/// @file linear/solvers/linear_solver.hpp
/// @brief Universal linear solver callable type.
#pragma once

#include "container/vector.hpp"
#include "linear/solvers/solver_result.hpp"
#include <functional>

namespace num {

/// @brief Callable that solves \f$Ax=\mathrm{rhs}\f$.
using linear_solver = std::function<solver_result(const vec &rhs, vec &x)>;

} // namespace num
