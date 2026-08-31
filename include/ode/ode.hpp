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

/// @brief Create a lazy forward Euler trajectory iterator for \f$\dot{y} = f(t, y)\f$.
///
/// @tparam RHS Callable type with signature `void(real t, const State& y, State& dy)`.
/// @tparam State State container type (defaults to `Vector`).
/// @param f Right-hand side derivative function.
/// @param y0 Initial state vector \f$y(t_0)\f$.
/// @param p Integration parameters (start time, end time, time step).
/// @return `BasicEulerSteps` lazy iterable sequence of integration steps.
template <typename RHS = ODERhsFn, typename State = Vector>
inline BasicEulerSteps<RHS, State> euler(RHS f, State y0, ODEParams p = {}) {
    return BasicEulerSteps<RHS, State>(std::move(f), std::move(y0), p);
}

/// @brief Create a lazy classical 4th-order Runge-Kutta (RK4) trajectory iterator.
///
/// Evaluates four intermediate stages per step: \f$k_1, k_2, k_3, k_4\f$ with \f$\mathcal{O}(h^4)\f$ local accuracy.
///
/// @tparam RHS Derivative callable `void(real t, const State& y, State& dy)`.
/// @tparam State State container type.
/// @param f Right-hand side derivative function.
/// @param y0 Initial state vector \f$y(t_0)\f$.
/// @param p Integration parameters (`t0`, `tf`, `dt`).
/// @return `BasicRK4Steps` lazy iterable sequence of integration steps.
template <typename RHS = ODERhsFn, typename State = Vector>
inline BasicRK4Steps<RHS, State> rk4(RHS f, State y0, ODEParams p = {}) {
    return BasicRK4Steps<RHS, State>(std::move(f), std::move(y0), p);
}

/// @brief Create a lazy adaptive Dormand-Prince (RK45) trajectory iterator with PI step control.
///
/// Uses embedded 4th and 5th order Runge-Kutta pairs (Dormand-Prince 5(4) with FSAL property)
/// to estimate truncation error and adaptively adjust time step \f$h\f$ to meet `rtol` and `atol`.
///
/// @tparam RHS Derivative callable `void(real t, const State& y, State& dy)`.
/// @tparam State State container type.
/// @param f Right-hand side derivative function.
/// @param y0 Initial state vector \f$y(t_0)\f$.
/// @param p Integration parameters (`t0`, `tf`, `rtol`, `atol`, `max_steps`).
/// @return `BasicRK45Steps` lazy iterable sequence of adaptive steps.
template <typename RHS = ODERhsFn, typename State = Vector>
inline BasicRK45Steps<RHS, State> rk45(RHS f, State y0, ODEParams p = {}) {
    return BasicRK45Steps<RHS, State>(std::move(f), std::move(y0), p);
}

/// @brief Create a lazy second-order velocity-Verlet trajectory iterator for Hamiltonian systems \f$\ddot{q} = a(q)\f$.
///
/// Exactly preserves symplectic 2-form \f$\mathrm{d}q \wedge \mathrm{d}p\f$ and bounds energy drift over long integration times.
///
/// @tparam Accel Acceleration callable `void(const State& q, State& a)`.
/// @tparam State Coordinate and velocity vector type.
/// @param accel Acceleration function \f$a(q) = -\nabla V(q)\f$.
/// @param q0 Initial generalized coordinates.
/// @param v0 Initial generalized velocities.
/// @param p Integration parameters (`t0`, `tf`, `dt`).
/// @return `BasicVerletSteps` lazy iterable sequence of symplectic steps.
template <typename Accel = AccelFn, typename State = Vector>
inline BasicVerletSteps<Accel, State> verlet(Accel accel, State q0, State v0, ODEParams p = {}) {
    return BasicVerletSteps<Accel, State>(std::move(accel), std::move(q0), std::move(v0), p);
}

/// @brief Create a lazy 4th-order symplectic Yoshida integrator trajectory for \f$\ddot{q} = a(q)\f$.
///
/// Uses symmetric composition of three velocity-Verlet sub-steps to achieve 4th-order symplectic accuracy.
///
/// @tparam Accel Acceleration callable `void(const State& q, State& a)`.
/// @tparam State Coordinate and velocity vector type.
/// @param accel Acceleration function \f$a(q)\f$.
/// @param q0 Initial coordinates.
/// @param v0 Initial velocities.
/// @param p Integration parameters (`t0`, `tf`, `dt`).
/// @return `BasicYoshida4Steps` lazy iterable sequence of 4th-order symplectic steps.
template <typename Accel = AccelFn, typename State = Vector>
inline BasicYoshida4Steps<Accel, State> yoshida4(Accel accel, State q0, State v0, ODEParams p = {}) {
    return BasicYoshida4Steps<Accel, State>(std::move(accel), std::move(q0), std::move(v0), p);
}

/// @brief Create a lazy 4th-order Nystrom integrator trajectory for second-order ODEs \f$\ddot{q} = a(q)\f$.
///
/// @tparam Accel Acceleration callable.
/// @tparam State State container type.
/// @param accel Acceleration function \f$a(q)\f$.
/// @param q0 Initial coordinates.
/// @param v0 Initial velocities.
/// @param p Integration parameters.
/// @return `BasicRK4_2ndSteps` lazy iterable sequence.
template <typename Accel = AccelFn, typename State = Vector>
inline BasicRK4_2ndSteps<Accel, State> rk4_2nd(Accel accel, State q0, State v0, ODEParams p = {}) {
    return BasicRK4_2ndSteps<Accel, State>(std::move(accel), std::move(q0), std::move(v0), p);
}

/// @brief Integrate first-order ODE \f$\dot{y} = f(t, y)\f$ using fixed-step forward Euler.
///
/// @tparam RHS Derivative callable.
/// @tparam State State vector type.
/// @param f Right-hand side function.
/// @param y0 Initial state \f$y(t_0)\f$.
/// @param p Integration parameters (`t0`, `tf`, `dt`).
/// @param observer Optional callback `void(real t, const Vector& u)` invoked at each step.
/// @return `ODEResult` with final state `u`, final time `t`, total `steps`, and convergence flag.
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

/// @brief Integrate first-order ODE \f$\dot{y} = f(t, y)\f$ using classical fixed-step RK4.
///
/// @tparam RHS Derivative callable.
/// @tparam State State vector type.
/// @param f Right-hand side function.
/// @param y0 Initial state \f$y(t_0)\f$.
/// @param p Integration parameters (`t0`, `tf`, `dt`).
/// @param observer Optional callback `void(real t, const Vector& u)` invoked at each step.
/// @return `ODEResult` with final state `u`, final time `t`, total `steps`, and convergence flag.
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

/// @brief Integrate first-order ODE \f$\dot{y} = f(t, y)\f$ using adaptive Dormand-Prince RK45.
///
/// @tparam RHS Derivative callable.
/// @tparam State State vector type.
/// @param f Right-hand side function.
/// @param y0 Initial state \f$y(t_0)\f$.
/// @param p Integration parameters (`t0`, `tf`, `rtol`, `atol`, `max_steps`).
/// @param observer Optional callback `void(real t, const Vector& u)` invoked at each step.
/// @return `ODEResult` with final state `u`, final time `t`, total `steps`, and convergence flag.
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

/// @brief Integrate second-order Hamiltonian system \f$\ddot{q} = a(q)\f$ using symplectic velocity Verlet.
///
/// @tparam Accel Acceleration callable.
/// @tparam State State vector type.
/// @param accel Acceleration function \f$a(q)\f$.
/// @param q0 Initial coordinates \f$q(t_0)\f$.
/// @param v0 Initial velocities \f$v(t_0)\f$.
/// @param p Integration parameters (`t0`, `tf`, `dt`).
/// @param observer Optional callback `void(real t, const Vector& q, const Vector& v)` invoked at each step.
/// @return `SymplecticResult` with final coordinates `q`, velocities `v`, time `t`, and `steps`.
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

/// @brief Integrate second-order Hamiltonian system \f$\ddot{q} = a(q)\f$ using 4th-order symplectic Yoshida splitting.
///
/// @tparam Accel Acceleration callable.
/// @tparam State State vector type.
/// @param accel Acceleration function \f$a(q)\f$.
/// @param q0 Initial coordinates \f$q(t_0)\f$.
/// @param v0 Initial velocities \f$v(t_0)\f$.
/// @param p Integration parameters (`t0`, `tf`, `dt`).
/// @param observer Optional callback `void(real t, const Vector& q, const Vector& v)` invoked at each step.
/// @return `SymplecticResult` with final coordinates `q`, velocities `v`, time `t`, and `steps`.
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

/// @brief Integrate second-order system \f$\ddot{q} = a(q)\f$ using 4th-order Nystrom Runge-Kutta.
///
/// @tparam Accel Acceleration callable.
/// @tparam State State vector type.
/// @param accel Acceleration function \f$a(q)\f$.
/// @param q0 Initial coordinates \f$q(t_0)\f$.
/// @param v0 Initial velocities \f$v(t_0)\f$.
/// @param p Integration parameters (`t0`, `tf`, `dt`).
/// @param observer Optional callback `void(real t, const Vector& q, const Vector& v)` invoked at each step.
/// @return `SymplecticResult` with final coordinates `q`, velocities `v`, time `t`, and `steps`.
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
