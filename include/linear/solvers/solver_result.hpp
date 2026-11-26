/// @file linear/solvers/solver_result.hpp
/// @brief Iterative solver result.
#pragma once
#include "core/types.hpp"

#include <ostream>

namespace num {

/// @brief Result of a linear solve: iterations, residual norm, convergence.
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
