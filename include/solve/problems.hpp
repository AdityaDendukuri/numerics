/// @file solve/problems.hpp
/// @brief Problem types: carry the mathematics, not the numerics.
#pragma once

#include "core/vector.hpp"
#include "ode/ode.hpp"
#include <concepts>

namespace num {

struct ODEProblem {
  ODERhsFn f;
  Vector u0;
  double t0 = 0.0;
  double tf = 1.0;
};

/// @brief Any type carrying an ODE right-hand side f(t,y,dy) and t0/tf/u0 data.
template<typename P>
concept IsODEProblem = requires(const P& p, double t, const Vector& y, Vector& dy) {
  p.f(t, y, dy);
  { p.u0 } -> std::convertible_to<const Vector&>;
  { p.t0 } -> std::convertible_to<double>;
  { p.tf } -> std::convertible_to<double>;
};

/// @brief Linear system A x = b. A is any matrix or LinearOperator; b the RHS.
/// Non-owning view over A and b (bind at the call site for an immediate solve).
template<class Op>
struct LinearProblem {
  const Op& A;
  const Vector& b;
};

template<class Op>
LinearProblem(const Op&, const Vector&) -> LinearProblem<Op>;

} // namespace num
