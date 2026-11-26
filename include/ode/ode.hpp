/// @file ode/ode.hpp
/// @brief ODE and symplectic integrators.
#pragma once

#include "ode/concepts.hpp"
#include "ode/debug.hpp"
#include "ode/implicit.hpp"
#include "ode/steps.hpp"
#include "ode/types.hpp"
#include <utility>

namespace num {

/// @brief Return the steps of the forward Euler method, computed as they are read:
/// \f$y_{n+1} = y_n + h f(t_n, y_n)\f$.
template <typename RHS = ode_rhs_fn, typename State = vec<real>>
requires vector_space<State> &&
    std::invocable<RHS &, real, const State &, State &> inline basic_euler_steps<RHS, State>
    euler(RHS f, State y0, ode_params p = {}) {
    return basic_euler_steps<RHS, State>(std::move(f), std::move(y0), p);
}

/// @brief Return the steps of the classical Runge–Kutta method (RK4), computed as they are read:
/// \f$\dot{y} = f(t, y)\f$.
template <typename RHS = ode_rhs_fn, typename State = vec<real>>
requires vector_space<State> &&
    std::invocable<RHS &, real, const State &, State &> inline basic_rk4_steps<RHS, State>
    rk4(RHS f, State y0, ode_params p = {}) {
    return basic_rk4_steps<RHS, State>(std::move(f), std::move(y0), p);
}

/// @brief Return the steps of the adaptive Dormand–Prince method (RK45), computed as they are read:
/// \f$\dot{y} = f(t, y)\f$.
template <typename RHS = ode_rhs_fn, typename State = vec<real>>
requires vector_space<State> &&
    std::invocable<RHS &, real, const State &, State &> inline basic_rk45_steps<RHS, State>
    rk45(RHS f, State y0, ode_params p = {}) {
    return basic_rk45_steps<RHS, State>(std::move(f), std::move(y0), p);
}

/// @brief Return the steps of the velocity Verlet method, computed as they are read:
/// \f$\ddot{q} = a(q)\f$.
/// @note Energy drift stays bounded over long runs.
template <typename Accel = accel_fn, typename State = vec<real>>
requires vector_space<State> &&
    std::invocable<Accel &, const State &, State &> inline basic_verlet_steps<Accel, State>
    verlet(Accel accel, State q0, State v0, ode_params p = {}) {
    return basic_verlet_steps<Accel, State>(std::move(accel), std::move(q0), std::move(v0), p);
}

/// @brief Return the steps of Yoshida's fourth-order method, computed as they are read:
/// \f$\ddot{q} = a(q)\f$.
template <typename Accel = accel_fn, typename State = vec<real>>
requires vector_space<State> &&
    std::invocable<Accel &, const State &, State &> inline basic_yoshida4_steps<Accel, State>
    yoshida4(Accel accel, State q0, State v0, ode_params p = {}) {
    return basic_yoshida4_steps<Accel, State>(std::move(accel), std::move(q0), std::move(v0), p);
}

/// @brief Return the steps of the Runge–Kutta–Nyström method (order 4), computed as they are read:
/// \f$\ddot{q} = a(q)\f$.
template <typename Accel = accel_fn, typename State = vec<real>>
requires vector_space<State> &&
    std::invocable<Accel &, const State &, State &> inline basic_rk4_2nd_steps<Accel, State>
    rk4_2nd(Accel accel, State q0, State v0, ode_params p = {}) {
    return basic_rk4_2nd_steps<Accel, State>(std::move(accel), std::move(q0), std::move(v0), p);
}

/// @brief Integrate by the forward Euler method: \f$y_{n+1} = y_n + h f(t_n, y_n)\f$.
/// `observer(t, u)` runs after each step.
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

/// @brief Integrate by the classical Runge–Kutta method (RK4): \f$\dot{y} = f(t, y)\f$.
/// `observer(t, u)` runs after each step.
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

/// @brief Integrate by the Dormand–Prince method (RK45), choosing the step size:
/// \f$\dot{y} = f(t, y)\f$.
/// `observer(t, u)` runs after each accepted step.
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

/// @brief Integrate by the velocity Verlet method, which is symplectic: \f$\ddot{q} = a(q)\f$.
/// `observer(t, q, v)` runs after each step.
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

/// @brief Integrate by Yoshida's fourth-order symplectic method: \f$\ddot{q} = a(q)\f$.
/// `observer(t, q, v)` runs after each step.
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

/// @brief Integrate by the fourth-order Runge–Kutta–Nyström method: \f$\ddot{q} = a(q)\f$.
/// `observer(t, q, v)` runs after each step.
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
