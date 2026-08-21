/// @file ode/concepts.hpp
/// @brief Compile-time contracts for ODE initial-value problems.
#pragma once

#include "ode/types.hpp"
#include <concepts>

namespace num {

/// Field exposing mutable vector storage to implicit time integrators.
template<class T>
concept VecField = requires(T& field) {
  { field.vec() } -> std::same_as<Vector&>;
};

/// Object carrying f(t,y,dy), an initial state, and a finite time interval.
template<typename Problem>
concept IsODEProblem =
  requires(const Problem& problem, real time, const Vector& state, Vector& derivative) {
    problem.f(time, state, derivative);
    { problem.u0 } -> std::convertible_to<const Vector&>;
    { problem.t0 } -> std::convertible_to<real>;
    { problem.tf } -> std::convertible_to<real>;
  };

} // namespace num
