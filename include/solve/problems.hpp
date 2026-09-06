/// @file solve/problems.hpp
/// @brief Problem types: carry the mathematics, not the numerics.
#pragma once

#include "container/vector.hpp"
#include "ode/concepts.hpp"

namespace num {

/// Initial-value problem y'=f(t,y) over [t0,tf].
struct ode_problem {
    ode_rhs_fn f;
    vec u0;
    double t0 = 0.0;
    double tf = 1.0;
};

/// @brief Linear system \f$A \mathbf{x} = \mathbf{b}\f$. \f$A\f$ is any matrix or `linear_operator`; \f$\mathbf{b}\f$ is the right-hand side.
/// Non-owning view over \f$A\f$ and \f$\mathbf{b}\f$ (bind at the call site for an immediate solve).
template <class Op>
struct linear_problem {
    const Op &A;
    const vec &b;
};

template <class Op>
linear_problem(const Op &, const vec &) -> linear_problem<Op>;

} // namespace num
