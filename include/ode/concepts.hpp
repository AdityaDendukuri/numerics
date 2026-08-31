/// @file ode/concepts.hpp
/// @brief Contracts for initial value problems and the state spaces they evolve in.
#pragma once

#include "container/concepts.hpp"
#include "ode/types.hpp"
#include <concepts>

namespace num {

/// @brief State exposing its underlying vector space to implicit integrators.
///
/// Implicit steppers solve a linear system in the state space, so they need the
/// state as a vector rather than as a field, grid or particle set.
template <class T>
concept VecField = requires(T &field) {
    { field.vec() } -> std::same_as<Vector &>;
};

/// @brief Initial value problem \f$\dot{y} = f(t,y)\f$ on \f$[t_0, t_f]\f$ with \f$y(t_0) = u_0\f$.
///
/// The mathematical content is a time-dependent vector field \f$f\f$ on a state
/// space, an initial state, and an interval. The state is required only to be a
/// `num::VectorSpace`, so a problem posed over complex amplitudes or over
/// single precision is as admissible as one over `double` — the earlier form fixed
/// the state to `Vector` and excluded both.
template <typename Problem, class State = Vector>
concept IsODEProblem = VectorSpace<State> &&
    requires(const Problem &problem, real time, const State &state, State &derivative) {
    problem.f(time, state, derivative);
    { problem.u0 } -> std::convertible_to<const State &>;
    { problem.t0 } -> std::convertible_to<real>;
    { problem.tf } -> std::convertible_to<real>;
};

/// @brief Separable Hamiltonian system \f$\dot q = f_{\mathrm{pos}}(t,p)\f$, \f$\dot p = f_{\mathrm{mom}}(t,q)\f$.
///
/// Separability is what symplectic integrators exploit: because each half-step
/// depends only on the *other* coordinate, the update is explicit and exactly
/// volume-preserving in phase space. A general Hamiltonian admits no such split,
/// which is why this is a distinct concept rather than a refinement of
/// `IsODEProblem`.
template <typename Problem, class State = Vector>
concept IsSymplecticODEProblem = VectorSpace<State> &&
    requires(const Problem &problem, real time, const State &q, const State &p, State &dq,
             State &dp) {
    problem.f_pos(time, p, dq);
    problem.f_mom(time, q, dp);
    { problem.q0 } -> std::convertible_to<const State &>;
    { problem.p0 } -> std::convertible_to<const State &>;
    { problem.t0 } -> std::convertible_to<real>;
    { problem.tf } -> std::convertible_to<real>;
};

/// @brief Stepper advancing a state from \f$t\f$ to \f$t + h\f$.
template <typename Stepper, class State = Vector>
concept IsODEStepper = VectorSpace<State> &&
    requires(Stepper &stepper, real t, real h, const State &y, State &y_next) {
    stepper.step(t, h, y, y_next);
};

} // namespace num
