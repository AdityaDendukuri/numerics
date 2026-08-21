/// @file ode/ode.hpp
/// @brief ODE and symplectic integrator entry points.
#pragma once

#include "ode/concepts.hpp"
#include "ode/implicit.hpp"
#include "ode/steps.hpp"
#include "ode/types.hpp"

namespace num {

/// Create a lazy forward Euler trajectory.
EulerSteps euler(ODERhsFn f, Vector y0, ODEParams p = {});
/// Create a lazy classical RK4 trajectory.
RK4Steps rk4(ODERhsFn f, Vector y0, ODEParams p = {});
/// Create a lazy adaptive Dormand-Prince trajectory.
RK45Steps rk45(ODERhsFn f, Vector y0, ODEParams p = {});

/// Create a lazy velocity-Verlet trajectory.
VerletSteps verlet(AccelFn accel, Vector q0, Vector v0, ODEParams p = {});
/// Create a lazy fourth-order Yoshida trajectory.
Yoshida4Steps yoshida4(AccelFn accel, Vector q0, Vector v0, ODEParams p = {});
/// Create a lazy fourth-order Nystrom trajectory.
RK4_2ndSteps rk4_2nd(AccelFn accel, Vector q0, Vector v0, ODEParams p = {});

/// Integrate with first-order fixed-step forward Euler.
ODEResult ode_euler(ODERhsFn f,
                    Vector y0,
                    ODEParams p = {},
                    const ObserverFn& observer = {});

/// Integrate with classical fourth-order fixed-step Runge-Kutta.
ODEResult ode_rk4(ODERhsFn f,
                  Vector y0,
                  ODEParams p = {},
                  const ObserverFn& observer = {});

/// Integrate with adaptive Dormand-Prince RK45 and PI step-size control.
ODEResult ode_rk45(ODERhsFn f,
                   Vector y0,
                   ODEParams p = {},
                   const ObserverFn& observer = {});

/// Integrate q''=a(q) with second-order symplectic velocity Verlet.
SymplecticResult ode_verlet(AccelFn accel,
                            Vector q0,
                            Vector v0,
                            ODEParams p = {},
                            const SympObserverFn& observer = {});

/// Integrate q''=a(q) with fourth-order symplectic Yoshida splitting.
SymplecticResult ode_yoshida4(AccelFn accel,
                              Vector q0,
                              Vector v0,
                              ODEParams p = {},
                              const SympObserverFn& observer = {});

/// Integrate q''=a(q) with non-symplectic fourth-order Nystrom RK4.
SymplecticResult ode_rk4_2nd(AccelFn accel,
                             Vector q0,
                             Vector v0,
                             ODEParams p = {},
                             const SympObserverFn& observer = {});

} // namespace num
