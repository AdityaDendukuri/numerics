/// @file linear/solvers/linear_solver.hpp
/// @brief Type-erased linear solver.
#pragma once

#include "container/vector.hpp"
#include "linear/solvers/solver_result.hpp"
#include <functional>

namespace num {

/// @brief Function that solves a linear system: \f$x \leftarrow A^{-1} b\f$.
using linear_solver = std::function<solver_result(const vec<real> &rhs, vec<real> &x)>;

} // namespace num
