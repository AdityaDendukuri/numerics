/// @file solver_result.hpp
/// @brief Common result type shared by all iterative solvers
#pragma once
#include "core/types.hpp"

#include <ostream>

namespace num {

/// Iteration count, final residual, and convergence status of a linear solve.
struct solver_result {
    idx iterations = 0;     ///< Number of iterations performed
    real residual = 0.0;    ///< Final residual norm ||b - Ax||
    bool converged = false; ///< Whether tolerance was met

    friend std::ostream &operator<<(std::ostream &os, const solver_result &r) {
        os << "solver_result{ converged: " << (r.converged ? "true" : "false")
           << ", iterations: " << r.iterations
           << ", residual: " << r.residual << " }";
        return os;
    }
};

} // namespace num
