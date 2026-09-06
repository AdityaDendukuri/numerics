/// @file ode/implicit.hpp
/// @brief Implicit time integration via a user-supplied linear_solver.
///
/// advance(u, solver, params)          -- fixed-step backward Euler, no
/// observer advance(u, solver, params, obs)     -- same, with step callback
///
/// The field type is constrained by the vec_field concept (ode/concepts.hpp):
/// any object exposing .as_vec() -> vec& works (vec itself, scalar_field_2d,
/// scalar_field_3d, ...). This keeps ode/ independent of the fields/ module while
/// supporting all types.
/// @todo Add Crank-Nicolson, BDF2, and IMEX step drivers with explicit mass
/// matrix/operator hooks.
#pragma once

#include "container/vector.hpp"
#include "linear/solvers/linear_solver.hpp"
#include "ode/concepts.hpp"

namespace num {
namespace ode {

/// Parameters for fixed-step implicit integration.
struct implicit_params {
    int nstep; ///< number of time steps
    double dt; ///< step size (reported to observer as t)
};

/// Advance u by nstep implicit steps using solver.
/// obs(step, t, u) is called at step 0 (initial) and after each solve.
template <vec_field field, typename Observer>
void advance(field &u, const linear_solver &solver, implicit_params p, Observer &&obs) {
    obs(0, 0.0, u);
    for (int s = 0; s < p.nstep; ++s) {
        vec rhs = u.as_vec();
        solver(rhs, u.as_vec());
        obs(s + 1, (s + 1) * p.dt, u);
    }
}

/// Overload without observer.
template <vec_field field>
void advance(field &u, const linear_solver &solver, implicit_params p) {
    for (int s = 0; s < p.nstep; ++s) {
        vec rhs = u.as_vec();
        solver(rhs, u.as_vec());
    }
}

} // namespace ode
} // namespace num
