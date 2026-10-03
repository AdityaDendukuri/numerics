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

/// @brief Lazy forward Euler trajectory for \f$\dot{y} = f(t, y)\f$.
///
/// `f` has the signature `void(real t, const State& y, State& dy)`, and `p` supplies `t0`, `tf`
/// and `dt`.
template <typename RHS = ode_rhs_fn, typename State = vec<real>>
requires vector_space<State> &&
    std::invocable<RHS &, real, const State &, State &> inline basic_euler_steps<RHS, State>
    euler(RHS f, State y0, ode_params p = {}) {
    return basic_euler_steps<RHS, State>(std::move(f), std::move(y0), p);
}

/// @brief Lazy classical fourth-order Runge--Kutta trajectory for \f$\dot{y} = f(t, y)\f$.
///
/// `f` has the signature `void(real t, const State& y, State& dy)`, and `p` supplies `t0`, `tf`
/// and `dt`.
template <typename RHS = ode_rhs_fn, typename State = vec<real>>
requires vector_space<State> &&
    std::invocable<RHS &, real, const State &, State &> inline basic_rk4_steps<RHS, State>
    rk4(RHS f, State y0, ode_params p = {}) {
    return basic_rk4_steps<RHS, State>(std::move(f), std::move(y0), p);
}

/// @brief Lazy adaptive Dormand--Prince 5(4) trajectory with PI step control.
///
/// The embedded pair estimates the local error and sets the step to meet `rtol` and `atol`.
/// `p` supplies `t0`, `tf`, `rtol`, `atol` and `max_steps`.
template <typename RHS = ode_rhs_fn, typename State = vec<real>>
requires vector_space<State> &&
    std::invocable<RHS &, real, const State &, State &> inline basic_rk45_steps<RHS, State>
    rk45(RHS f, State y0, ode_params p = {}) {
    return basic_rk45_steps<RHS, State>(std::move(f), std::move(y0), p);
}

/// @brief Lazy velocity-Verlet trajectory for \f$\ddot{q} = a(q)\f$.
///
/// Symplectic, so energy drift stays bounded over long runs. `accel` has the signature
/// `void(const State& q, State& a)`.
template <typename Accel = accel_fn, typename State = vec<real>>
requires vector_space<State> &&
    std::invocable<Accel &, const State &, State &> inline basic_verlet_steps<Accel, State>
    verlet(Accel accel, State q0, State v0, ode_params p = {}) {
    return basic_verlet_steps<Accel, State>(std::move(accel), std::move(q0), std::move(v0), p);
}

/// @brief Lazy fourth-order Yoshida trajectory for \f$\ddot{q} = a(q)\f$.
///
/// A symmetric composition of three velocity-Verlet substeps, so it stays symplectic.
template <typename Accel = accel_fn, typename State = vec<real>>
requires vector_space<State> &&
    std::invocable<Accel &, const State &, State &> inline basic_yoshida4_steps<Accel, State>
    yoshida4(Accel accel, State q0, State v0, ode_params p = {}) {
    return basic_yoshida4_steps<Accel, State>(std::move(accel), std::move(q0), std::move(v0), p);
}

/// @brief Lazy fourth-order Nystrom trajectory for \f$\ddot{q} = a(q)\f$.
template <typename Accel = accel_fn, typename State = vec<real>>
requires vector_space<State> &&
    std::invocable<Accel &, const State &, State &> inline basic_rk4_2nd_steps<Accel, State>
    rk4_2nd(Accel accel, State q0, State v0, ode_params p = {}) {
    return basic_rk4_2nd_steps<Accel, State>(std::move(accel), std::move(q0), std::move(v0), p);
}

/// @brief Integrate \f$\dot{y} = f(t, y)\f$ with fixed-step forward Euler.
///
/// `observer`, if given, is called as `observer(t, u)` after each step.
template <typename RHS = ode_rhs_fn, typename State = vec<real>>
requires vector_space<State> &&std::invocable<RHS &, real, const State &, State &> inline ode_result
ode_euler(RHS f, State y0, ode_params p = {}, const observer_fn &observer = {}) {
    auto s = euler(std::move(f), std::move(y0), p);
    if (!observer) {
        return s.run();
    }
    for (auto step : s) {
        observer(step.t, step.u);
    }
    return s.run();
}

/// @brief Integrate \f$\dot{y} = f(t, y)\f$ with fixed-step classical RK4.
///
/// `observer`, if given, is called as `observer(t, u)` after each step.
template <typename RHS = ode_rhs_fn, typename State = vec<real>>
requires vector_space<State> &&std::invocable<RHS &, real, const State &, State &> inline ode_result
ode_rk4(RHS f, State y0, ode_params p = {}, const observer_fn &observer = {}) {
    auto s = rk4(std::move(f), std::move(y0), p);
    if (!observer) {
        return s.run();
    }
    for (auto step : s) {
        observer(step.t, step.u);
    }
    return s.run();
}

/// @brief Integrate \f$\dot{y} = f(t, y)\f$ with adaptive Dormand--Prince RK45.
///
/// `observer`, if given, is called as `observer(t, u)` after each accepted step.
template <typename RHS = ode_rhs_fn, typename State = vec<real>>
requires vector_space<State> &&std::invocable<RHS &, real, const State &, State &> inline ode_result
ode_rk45(RHS f, State y0, ode_params p = {}, const observer_fn &observer = {}) {
    auto s = rk45(std::move(f), std::move(y0), p);
    if (!observer) {
        return s.run();
    }
    for (auto step : s) {
        observer(step.t, step.u);
    }
    return s.run();
}

/// @brief Integrate \f$\ddot{q} = a(q)\f$ with velocity Verlet.
///
/// `observer`, if given, is called as `observer(t, q, v)` after each step.
template <typename Accel = accel_fn, typename State = vec<real>>
requires vector_space<State> &&
    std::invocable<Accel &, const State &, State &> inline symplectic_result
    ode_verlet(Accel accel, State q0, State v0, ode_params p = {},
               const symp_observer_fn &observer = {}) {
    auto s = verlet(std::move(accel), std::move(q0), std::move(v0), p);
    if (!observer) {
        return s.run();
    }
    for (auto step : s) {
        observer(step.t, step.q, step.v);
    }
    return s.run();
}

/// @brief Integrate \f$\ddot{q} = a(q)\f$ with fourth-order Yoshida splitting.
///
/// `observer`, if given, is called as `observer(t, q, v)` after each step.
template <typename Accel = accel_fn, typename State = vec<real>>
requires vector_space<State> &&
    std::invocable<Accel &, const State &, State &> inline symplectic_result
    ode_yoshida4(Accel accel, State q0, State v0, ode_params p = {},
                 const symp_observer_fn &observer = {}) {
    auto s = yoshida4(std::move(accel), std::move(q0), std::move(v0), p);
    if (!observer) {
        return s.run();
    }
    for (auto step : s) {
        observer(step.t, step.q, step.v);
    }
    return s.run();
}

/// @brief Integrate \f$\ddot{q} = a(q)\f$ with fourth-order Nystrom Runge--Kutta.
///
/// `observer`, if given, is called as `observer(t, q, v)` after each step.
template <typename Accel = accel_fn, typename State = vec<real>>
requires vector_space<State> &&
    std::invocable<Accel &, const State &, State &> inline symplectic_result
    ode_rk4_2nd(Accel accel, State q0, State v0, ode_params p = {},
                const symp_observer_fn &observer = {}) {
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
