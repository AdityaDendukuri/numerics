/// @file ode/concepts.hpp
/// @brief Compile-time contracts for ODE initial-value problems and steppers.
#pragma once

#include "ode/types.hpp"
#include <concepts>

namespace num {

/// @brief Field exposing mutable vector storage to implicit time integrators.
template <class T>
concept VecField = requires(T &field) {
    { field.vec() } -> std::same_as<Vector &>;
};

/// @brief Object carrying f(t,y,dy), an initial state, and a finite time interval.
template <typename Problem>
concept IsODEProblem =
    requires(const Problem &problem, real time, const Vector &state, Vector &derivative) {
    problem.f(time, state, derivative);
    { problem.u0 } -> std::convertible_to<const Vector &>;
    { problem.t0 } -> std::convertible_to<real>;
    { problem.tf } -> std::convertible_to<real>;
};

/// @brief Separable Hamiltonian or second-order system for symplectic integrators.
template <typename Problem>
concept IsSymplecticODEProblem =
    requires(const Problem &problem, real time, const Vector &q, const Vector &p, Vector &dq,
             Vector &dp) {
    problem.f_pos(time, p, dq);
    problem.f_mom(time, q, dp);
    { problem.q0 } -> std::convertible_to<const Vector &>;
    { problem.p0 } -> std::convertible_to<const Vector &>;
    { problem.t0 } -> std::convertible_to<real>;
    { problem.tf } -> std::convertible_to<real>;
};

} // namespace num
