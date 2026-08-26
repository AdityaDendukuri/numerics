/// @file ode/ode.hpp
/// @brief ODE and symplectic integrator entry points.
#pragma once

#include "ode/concepts.hpp"
#include "ode/debug.hpp"
#include "ode/implicit.hpp"
#include "ode/steps.hpp"
#include "ode/types.hpp"
#include <utility>

namespace num {

/// Create a lazy forward Euler trajectory.
template <typename RHS = ODERhsFn, typename State = Vector>
inline BasicEulerSteps<RHS, State> euler(RHS f, State y0, ODEParams p = {}) {
    return BasicEulerSteps<RHS, State>(std::move(f), std::move(y0), p);
}

/// Create a lazy classical RK4 trajectory.
template <typename RHS = ODERhsFn, typename State = Vector>
inline BasicRK4Steps<RHS, State> rk4(RHS f, State y0, ODEParams p = {}) {
    return BasicRK4Steps<RHS, State>(std::move(f), std::move(y0), p);
}

/// Create a lazy adaptive Dormand-Prince trajectory.
template <typename RHS = ODERhsFn, typename State = Vector>
inline BasicRK45Steps<RHS, State> rk45(RHS f, State y0, ODEParams p = {}) {
    return BasicRK45Steps<RHS, State>(std::move(f), std::move(y0), p);
}

/// Create a lazy velocity-Verlet trajectory.
template <typename Accel = AccelFn, typename State = Vector>
inline BasicVerletSteps<Accel, State> verlet(Accel accel, State q0, State v0, ODEParams p = {}) {
    return BasicVerletSteps<Accel, State>(std::move(accel), std::move(q0), std::move(v0), p);
}

/// Create a lazy fourth-order Yoshida trajectory.
template <typename Accel = AccelFn, typename State = Vector>
inline BasicYoshida4Steps<Accel, State> yoshida4(Accel accel, State q0, State v0, ODEParams p = {}) {
    return BasicYoshida4Steps<Accel, State>(std::move(accel), std::move(q0), std::move(v0), p);
}

/// Create a lazy fourth-order Nystrom trajectory.
template <typename Accel = AccelFn, typename State = Vector>
inline BasicRK4_2ndSteps<Accel, State> rk4_2nd(Accel accel, State q0, State v0, ODEParams p = {}) {
    return BasicRK4_2ndSteps<Accel, State>(std::move(accel), std::move(q0), std::move(v0), p);
}

/// Integrate with first-order fixed-step forward Euler.
template <typename RHS = ODERhsFn, typename State = Vector>
inline ODEResult ode_euler(RHS f, State y0, ODEParams p = {}, const ObserverFn &observer = {}) {
    auto s = euler(std::move(f), std::move(y0), p);
    if (!observer) {
        return s.run();
    }
    for (auto step : s) {
        observer(step.t, step.u);
    }
    return s.run();
}

/// Integrate with classical fourth-order fixed-step Runge-Kutta.
template <typename RHS = ODERhsFn, typename State = Vector>
inline ODEResult ode_rk4(RHS f, State y0, ODEParams p = {}, const ObserverFn &observer = {}) {
    auto s = rk4(std::move(f), std::move(y0), p);
    if (!observer) {
        return s.run();
    }
    for (auto step : s) {
        observer(step.t, step.u);
    }
    return s.run();
}

/// Integrate with adaptive Dormand-Prince RK45 and PI step-size control.
template <typename RHS = ODERhsFn, typename State = Vector>
inline ODEResult ode_rk45(RHS f, State y0, ODEParams p = {}, const ObserverFn &observer = {}) {
    auto s = rk45(std::move(f), std::move(y0), p);
    if (!observer) {
        return s.run();
    }
    for (auto step : s) {
        observer(step.t, step.u);
    }
    return s.run();
}

/// Integrate q''=a(q) with second-order symplectic velocity Verlet.
template <typename Accel = AccelFn, typename State = Vector>
inline SymplecticResult ode_verlet(Accel accel, State q0, State v0, ODEParams p = {},
                                   const SympObserverFn &observer = {}) {
    auto s = verlet(std::move(accel), std::move(q0), std::move(v0), p);
    if (!observer) {
        return s.run();
    }
    for (auto step : s) {
        observer(step.t, step.q, step.v);
    }
    return s.run();
}

/// Integrate q''=a(q) with fourth-order symplectic Yoshida splitting.
template <typename Accel = AccelFn, typename State = Vector>
inline SymplecticResult ode_yoshida4(Accel accel, State q0, State v0, ODEParams p = {},
                                     const SympObserverFn &observer = {}) {
    auto s = yoshida4(std::move(accel), std::move(q0), std::move(v0), p);
    if (!observer) {
        return s.run();
    }
    for (auto step : s) {
        observer(step.t, step.q, step.v);
    }
    return s.run();
}

/// Integrate q''=a(q) with non-symplectic fourth-order Nystrom RK4.
template <typename Accel = AccelFn, typename State = Vector>
inline SymplecticResult ode_rk4_2nd(Accel accel, State q0, State v0, ODEParams p = {},
                                    const SympObserverFn &observer = {}) {
    auto s = rk4_2nd(std::move(accel), std::move(q0), std::move(v0), p);
    if (!observer) {
        return s.run();
    }
    for (auto step : s) {
        observer(step.t, step.q, step.v);
    }
    return s.run();
}

} // namespace num
