/// @file ode/implicit.hpp
/// @brief Fixed-step backward Euler through a user-supplied linear_solver.
///
/// `advance(u, solver, params)` and `advance(u, solver, params, obs)`, the latter with a step
/// callback. The field is any `vec_field`, meaning anything with `.as_vec()`.
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
