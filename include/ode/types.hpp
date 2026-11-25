/// @file ode/types.hpp
/// @brief ODE callbacks, parameters, snapshots and results.
#pragma once

#include "container/vector.hpp"
#include <functional>

namespace num {

/// @brief Right-hand side: writes \f$\dot{y} = f(t, y)\f$.
using ode_rhs_fn = std::function<void(real t, const vec<real> &y, vec<real> &dydt)>;
/// @brief Acceleration: writes \f$a(q)\f$ for \f$\ddot{q} = a(q)\f$.
using accel_fn = std::function<void(const vec<real> &q, vec<real> &acc)>;
/// @brief First-order step observer: `(t, y)`.
using observer_fn = std::function<void(real t, const vec<real> &y)>;
/// @brief Second-order step observer: `(t, q, v)`.
using symp_observer_fn = std::function<void(real t, const vec<real> &q, const vec<real> &v)>;

/// @brief First-order step snapshot: \f$(t, y)\f$.
struct ode_step {
    real t = 0.0;
    vec<real> u;
};

/// @brief Second-order step snapshot: \f$(t, q, v)\f$.
struct symplectic_step {
    real t = 0.0;
    vec<real> q;
    vec<real> v;
};

#include <ostream>

/// @brief First-order result: final state and step counts.
struct ode_result {
    vec<real> u;
    real t = 0.0;
    idx steps = 0;
    bool converged = false;

    friend std::ostream &operator<<(std::ostream &os, const ode_result &r) {
        os << "ode_result{ t: " << r.t
           << ", steps: " << r.steps
           << ", converged: " << (r.converged ? "true" : "false")
           << ", u: [" << r.u.size() << " elements] }";
        return os;
    }
};

/// @brief Second-order result: final position and velocity.
struct symplectic_result {
    vec<real> q;
    vec<real> v;
    real t = 0.0;
    idx steps = 0;

    friend std::ostream &operator<<(std::ostream &os, const symplectic_result &r) {
        os << "symplectic_result{ t: " << r.t
           << ", steps: " << r.steps
           << ", q: [" << r.q.size() << " elements]"
           << ", v: [" << r.v.size() << " elements] }";
        return os;
    }
};

/// @brief Integration parameters: interval, step, tolerances, step limit.
struct ode_params {
    real t0 = 0.0;
    real tf = 1.0;
    real h = 1e-3;
    real rtol = 1e-6;
    real atol = 1e-9;
    idx max_steps = 1000000;
};

/// @brief End sentinel of a step range.
struct step_end {};

} // namespace num
