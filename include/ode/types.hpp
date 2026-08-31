/// @file ode/types.hpp
/// @brief Shared callbacks, parameters, snapshots, and results for ODE solvers.
#pragma once

#include "container/vector.hpp"
#include <functional>

namespace num {

/// First-order system callback that writes dy/dt for (t,y).
using ODERhsFn = std::function<void(real t, const Vector &y, Vector &dydt)>;
/// Position-dependent acceleration callback for second-order systems.
using AccelFn = std::function<void(const Vector &q, Vector &acc)>;
/// Optional callback invoked after accepted first-order steps.
using ObserverFn = std::function<void(real t, const Vector &y)>;
/// Optional callback invoked after accepted symplectic steps.
using SympObserverFn = std::function<void(real t, const Vector &q, const Vector &v)>;

/// Snapshot yielded by a first-order lazy integrator.
struct Step {
    real t = 0.0;
    Vector u;
};

/// Snapshot yielded by a second-order lazy integrator.
struct SymplecticStep {
    real t = 0.0;
    Vector q;
    Vector v;
};

#include <ostream>

/// Final state and convergence metadata for a first-order integration.
struct ODEResult {
    Vector u;
    real t = 0.0;
    idx steps = 0;
    bool converged = false;

    friend std::ostream &operator<<(std::ostream &os, const ODEResult &r) {
        os << "ODEResult{ t: " << r.t
           << ", steps: " << r.steps
           << ", converged: " << (r.converged ? "true" : "false")
           << ", u: [" << r.u.size() << " elements] }";
        return os;
    }
};

/// Final position and velocity from a second-order integration.
struct SymplecticResult {
    Vector q;
    Vector v;
    real t = 0.0;
    idx steps = 0;

    friend std::ostream &operator<<(std::ostream &os, const SymplecticResult &r) {
        os << "SymplecticResult{ t: " << r.t
           << ", steps: " << r.steps
           << ", q: [" << r.q.size() << " elements]"
           << ", v: [" << r.v.size() << " elements] }";
        return os;
    }
};

/// Shared integration interval, step-size, tolerance, and work limits.
struct ODEParams {
    real t0 = 0.0;
    real tf = 1.0;
    real h = 1e-3;
    real rtol = 1e-6;
    real atol = 1e-9;
    idx max_steps = 1000000;
};

/// Sentinel marking the end of a lazy integration range.
struct StepEnd {};

} // namespace num
