/// @file solver_result.hpp
/// @brief Common result type shared by all iterative solvers
#pragma once
#include "core/types.hpp"

namespace num {

struct SolverResult {
    idx  iterations;   ///< Number of iterations performed
    real residual;     ///< Final residual norm ||b - Ax||
    bool converged;    ///< Whether tolerance was met
};

} // namespace num
