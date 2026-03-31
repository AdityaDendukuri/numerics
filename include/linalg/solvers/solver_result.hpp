/// @file solver_result.hpp
/// @brief Common result type shared by all iterative solvers
#pragma once
#include "core/types.hpp"

namespace num {

struct SolverResult {
    idx  iterations = 0;     ///< Number of iterations performed
    real residual   = 0.0;   ///< Final residual norm ||b - Ax||
    bool converged  = false; ///< Whether tolerance was met
};

} // namespace num
