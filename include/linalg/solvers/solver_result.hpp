/// @file solver_result.hpp
/// @brief Common result type shared by all iterative solvers
#pragma once
#include "core/types.hpp"

namespace num {

/// Iteration count, final residual, and convergence status of a linear solve.
struct SolverResult {
  idx iterations = 0; ///< Number of iterations performed
  real residual = 0.0; ///< Final residual norm ||b - Ax||
  bool converged = false; ///< Whether tolerance was met
};

} // namespace num
