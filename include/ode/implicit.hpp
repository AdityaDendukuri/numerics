/// @file ode/implicit.hpp
/// @brief Implicit time integration via a user-supplied LinearSolver.
///
/// advance(u, solver, params)          -- fixed-step backward Euler, no
/// observer advance(u, solver, params, obs)     -- same, with step callback
///
/// The field type is constrained by the VecField concept (core/concepts.hpp):
/// any object exposing .vec() -> Vector& works (Vector itself, ScalarField2D,
/// ScalarField3D, ...). This keeps ode/ independent of the fields/ module while
/// supporting all types.
/// @todo Add Crank-Nicolson, BDF2, and IMEX step drivers with explicit mass
/// matrix/operator hooks.
#pragma once

#include "core/concepts.hpp"
#include "core/vector.hpp"
#include "linalg/solvers/linear_solver.hpp"

namespace num {
namespace ode {

/// Parameters for fixed-step implicit integration.
struct ImplicitParams {
  int nstep; ///< number of time steps
  double dt; ///< step size (reported to observer as t)
};

/// Advance u by nstep implicit steps using solver.
/// obs(step, t, u) is called at step 0 (initial) and after each solve.
template<VecField Field, typename Observer>
void advance(Field& u, const LinearSolver& solver, ImplicitParams p, Observer&& obs) {
  obs(0, 0.0, u);
  for (int s = 0; s < p.nstep; ++s) {
    Vector rhs = u.vec();
    solver(rhs, u.vec());
    obs(s + 1, (s + 1) * p.dt, u);
  }
}

/// Overload without observer.
template<VecField Field>
void advance(Field& u, const LinearSolver& solver, ImplicitParams p) {
  for (int s = 0; s < p.nstep; ++s) {
    Vector rhs = u.vec();
    solver(rhs, u.vec());
  }
}

} // namespace ode
} // namespace num
