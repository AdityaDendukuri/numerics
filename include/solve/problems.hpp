/// @file solve/problems.hpp
/// @brief Problem types: carry the mathematics, not the numerics.
#pragma once

#include "core/vector.hpp"
#include "ode/concepts.hpp"

namespace num {

/// Initial-value problem y'=f(t,y) over [t0,tf].
struct ODEProblem {
    ODERhsFn f;
    Vector u0;
    double t0 = 0.0;
    double tf = 1.0;
};

/// @brief Linear system A x = b. A is any matrix or LinearOperator; b the RHS.
/// Non-owning view over A and b (bind at the call site for an immediate solve).
template <class Op>
struct LinearProblem {
    const Op &A;
    const Vector &b;
};

template <class Op>
LinearProblem(const Op &, const Vector &) -> LinearProblem<Op>;

} // namespace num
