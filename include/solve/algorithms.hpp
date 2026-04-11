/// @file solve/algorithms.hpp
/// @brief Algorithm tags: carry the numerics, not the mathematics.
///
/// Explicit ODE (use with ODEProblem):
///   Euler        -- forward Euler,  O(h)
///   RK4          -- classic 4th-order Runge-Kutta
///   RK45         -- adaptive Dormand-Prince, O(h^5)
///
/// Implicit ODE (use with any VecField):
///   BackwardEuler -- fixed-step implicit Euler via a LinearSolver
///
/// MCMC (use with MCMCProblem):
///   Metropolis   -- Metropolis-Hastings with equilibration + measurement phases
#pragma once

#include "linalg/solvers/linear_solver.hpp"
#include "core/types.hpp"

namespace num {

// --- Explicit ODE algorithms -------------------------------------------------

struct Euler {
    double h = 1e-3;
};

struct RK4 {
    double h = 1e-3;
};

struct RK45 {
    double h         = 1e-3;
    double rtol      = 1e-6;
    double atol      = 1e-9;
    idx    max_steps = 1000000;
};

// --- Implicit ODE algorithms -------------------------------------------------

struct BackwardEuler {
    LinearSolver solver;
    double       dt;
    int          nstep;
};

// --- MCMC algorithms ---------------------------------------------------------

struct Metropolis {
    int equilibration = 1000; ///< sweeps discarded before measuring
    int measurements  = 500;  ///< sweeps over which the observable is averaged
};

} // namespace num
