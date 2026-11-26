/// @file ode/implicit.hpp
/// @brief Fixed-step backward Euler through a user-supplied linear solver.
// TODO: Crank-Nicolson, BDF2 and IMEX drivers with mass-matrix hooks.
#pragma once

#include "container/vector.hpp"
#include "linear/solvers/linear_solver.hpp"
#include "ode/concepts.hpp"

namespace num {
namespace ode {

/// @brief Settings for implicit integration with a fixed step.
struct implicit_params {
    int nstep; ///< number of time steps
    double dt; ///< step size (reported to observer as t)
};

/// @brief Integrate by the backward Euler method for `nstep` steps:
/// \f$(I - \Delta t\, L) u_{n+1} = u_n\f$.
/// `obs(step, t, u)` runs at step 0 and after each solve.
template <vec_field field, typename Observer>
void advance(field &u, const linear_solver &solver, implicit_params p, Observer &&obs) {
    obs(0, 0.0, u);
    for (int s = 0; s < p.nstep; ++s) {
        vec<real> rhs = u.as_vec();
        solver(rhs, u.as_vec());
        obs(s + 1, (s + 1) * p.dt, u);
    }
}

/// @brief Integrate by the backward Euler method for `nstep` steps:
/// \f$(I - \Delta t\, L) u_{n+1} = u_n\f$.
template <vec_field field>
void advance(field &u, const linear_solver &solver, implicit_params p) {
    for (int s = 0; s < p.nstep; ++s) {
        vec<real> rhs = u.as_vec();
        solver(rhs, u.as_vec());
    }
}

} // namespace ode
} // namespace num
