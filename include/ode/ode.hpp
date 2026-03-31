/// @file ode/ode.hpp
/// @brief ODE integrators: Euler, RK4, adaptive RK45 (Dormand-Prince), and
///        symplectic Velocity Verlet / Yoshida-4 for Hamiltonian systems.
///
/// General ODE solvers advance \f$ \dot{y} = f(t,y) \f$:
///   - ode_euler   -- forward Euler, O(h)
///   - ode_rk4     -- classic 4th-order Runge-Kutta, O(h^4)
///   - ode_rk45    -- adaptive Dormand-Prince with PI step control, O(h^5)
///
/// Symplectic integrators for separable \f$ H(q,p) = T(p) + V(q) \f$:
///   - ode_verlet   -- velocity Verlet, 2nd-order symplectic, 1 force eval/step
///   - ode_yoshida4 -- Yoshida (1990) 4th-order symplectic, 3 force evals/step
///   - ode_rk4_2nd  -- non-symplectic RK4 in Nystrom form, for comparison
///
/// Each integrator has a corresponding lazy-range factory (rk4, rk45, verlet,
/// etc.). Iterate with a range-based for to observe intermediate states, or
/// call .run() / use the ode_*() wrappers to obtain only the final result.
#pragma once

#include "core/types.hpp"
#include "core/vector.hpp"
#include <functional>

namespace num {

using ODERhsFn = std::function<void(real t, const Vector& y, Vector& dydt)>;
using AccelFn  = std::function<void(const Vector& q, Vector& acc)>;

struct Step {
    real   t;
    Vector y;
};

struct SymplecticStep {
    real   t;
    Vector q;
    Vector v;
};

struct ODEResult {
    Vector y;
    real   t;
    idx    steps;
    bool   converged;
};

struct SymplecticResult {
    Vector q;
    Vector v;
    real   t;
    idx    steps;
};

struct StepEnd {};

class EulerSteps {
    ODERhsFn f_;
    Vector   y_, dydt_;
    real     t_, t1_, h_;
    idx      steps_ = 0;
    bool     done_  = false;

    void advance();

public:
    explicit EulerSteps(ODERhsFn f, Vector y0, real t0, real t1, real h);

    struct iterator {
        EulerSteps* owner_;
        Step      operator*()  const { return {owner_->t_, owner_->y_}; }
        iterator& operator++()       { owner_->advance(); return *this; }
        bool operator!=(StepEnd) const { return !owner_->done_; }
        bool operator==(StepEnd) const { return  owner_->done_; }
    };

    iterator begin() { advance(); return {this}; }
    StepEnd  end()   const { return {}; }
    ODEResult run();
};

class RK4Steps {
    ODERhsFn f_;
    Vector   y_, k1_, k2_, k3_, k4_, ytmp_;
    real     t_, t1_, h_;
    idx      steps_ = 0;
    bool     done_  = false;

    void advance();

public:
    explicit RK4Steps(ODERhsFn f, Vector y0, real t0, real t1, real h);

    struct iterator {
        RK4Steps* owner_;
        Step      operator*()  const { return {owner_->t_, owner_->y_}; }
        iterator& operator++()       { owner_->advance(); return *this; }
        bool operator!=(StepEnd) const { return !owner_->done_; }
        bool operator==(StepEnd) const { return  owner_->done_; }
    };

    iterator begin() { advance(); return {this}; }
    StepEnd  end()   const { return {}; }
    ODEResult run();
};

class RK45Steps {
    ODERhsFn f_;
    Vector   y_, k1_, k2_, k3_, k4_, k5_, k6_, k7_, ytmp_, err_;
    real     t_, t1_, h_, rtol_, atol_;
    idx      steps_ = 0, max_steps_;
    bool     done_ = false, converged_ = true;

    void advance();

public:
    explicit RK45Steps(ODERhsFn f, Vector y0, real t0, real t1,
                       real rtol, real atol, real h0, idx max_steps);

    struct iterator {
        RK45Steps* owner_;
        Step       operator*()  const { return {owner_->t_, owner_->y_}; }
        iterator&  operator++()       { owner_->advance(); return *this; }
        bool operator!=(StepEnd) const { return !owner_->done_; }
        bool operator==(StepEnd) const { return  owner_->done_; }
    };

    iterator begin() { advance(); return {this}; }
    StepEnd  end()   const { return {}; }
    ODEResult run();
};

class VerletSteps {
    AccelFn accel_;
    Vector  q_, v_, a_cur_, a_next_;
    real    t_, t1_, h_;
    idx     steps_ = 0;
    bool    done_  = false;

    void advance();

public:
    explicit VerletSteps(AccelFn accel, Vector q0, Vector v0,
                         real t0, real t1, real h);

    struct iterator {
        VerletSteps*   owner_;
        SymplecticStep operator*()  const { return {owner_->t_, owner_->q_, owner_->v_}; }
        iterator&      operator++()       { owner_->advance(); return *this; }
        bool operator!=(StepEnd) const { return !owner_->done_; }
        bool operator==(StepEnd) const { return  owner_->done_; }
    };

    iterator begin() { advance(); return {this}; }
    StepEnd  end()   const { return {}; }
    SymplecticResult run();
};

class Yoshida4Steps {
    AccelFn accel_;
    Vector  q_, v_, acc_;
    real    t_, t1_, h_;
    idx     steps_ = 0;
    bool    done_  = false;

    void advance();

public:
    explicit Yoshida4Steps(AccelFn accel, Vector q0, Vector v0,
                           real t0, real t1, real h);

    struct iterator {
        Yoshida4Steps* owner_;
        SymplecticStep operator*()  const { return {owner_->t_, owner_->q_, owner_->v_}; }
        iterator&      operator++()       { owner_->advance(); return *this; }
        bool operator!=(StepEnd) const { return !owner_->done_; }
        bool operator==(StepEnd) const { return  owner_->done_; }
    };

    iterator begin() { advance(); return {this}; }
    StepEnd  end()   const { return {}; }
    SymplecticResult run();
};

class RK4_2ndSteps {
    AccelFn accel_;
    Vector  q_, v_, a1_, a2_, a3_, a4_, qtmp_;
    real    t_, t1_, h_;
    idx     steps_ = 0;
    bool    done_  = false;

    void advance();

public:
    explicit RK4_2ndSteps(AccelFn accel, Vector q0, Vector v0,
                          real t0, real t1, real h);

    struct iterator {
        RK4_2ndSteps*  owner_;
        SymplecticStep operator*()  const { return {owner_->t_, owner_->q_, owner_->v_}; }
        iterator&      operator++()       { owner_->advance(); return *this; }
        bool operator!=(StepEnd) const { return !owner_->done_; }
        bool operator==(StepEnd) const { return  owner_->done_; }
    };

    iterator begin() { advance(); return {this}; }
    StepEnd  end()   const { return {}; }
    SymplecticResult run();
};

// Lazy-range factories

EulerSteps    euler   (ODERhsFn f, Vector y0, real t0, real t1, real h);
RK4Steps      rk4     (ODERhsFn f, Vector y0, real t0, real t1, real h);
RK45Steps     rk45    (ODERhsFn f, Vector y0, real t0, real t1,
                       real rtol = 1e-6, real atol = 1e-9,
                       real h0 = 1e-3, idx max_steps = 1000000);

VerletSteps   verlet  (AccelFn accel, Vector q0, Vector v0, real t0, real t1, real h);
Yoshida4Steps yoshida4(AccelFn accel, Vector q0, Vector v0, real t0, real t1, real h);
RK4_2ndSteps  rk4_2nd (AccelFn accel, Vector q0, Vector v0, real t0, real t1, real h);

// High-level integrators — return final state only

/// @brief Forward Euler, 1st-order, fixed step.
///
/// \f[ y_{n+1} = y_n + h\,f(t_n,\,y_n) \f]
///
/// Local truncation error \f$ O(h^2) \f$; global error \f$ O(h) \f$.
ODEResult ode_euler(ODERhsFn f, Vector y0, real t0, real t1, real h);

/// @brief Classic 4th-order Runge-Kutta, fixed step.
///
/// Butcher tableau (explicit, 4-stage):
/// \f[
///   k_1 = f(t_n,\,y_n), \quad
///   k_2 = f\!\left(t_n+\tfrac{h}{2},\,y_n+\tfrac{h}{2}k_1\right),
/// \f]
/// \f[
///   k_3 = f\!\left(t_n+\tfrac{h}{2},\,y_n+\tfrac{h}{2}k_2\right), \quad
///   k_4 = f(t_n+h,\,y_n+hk_3),
/// \f]
/// \f[
///   y_{n+1} = y_n + \tfrac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4).
/// \f]
///
/// Global error \f$ O(h^4) \f$; not symplectic (energy drifts for Hamiltonians).
ODEResult ode_rk4(ODERhsFn f, Vector y0, real t0, real t1, real h);

/// @brief Adaptive Dormand-Prince RK45 with FSAL and PI step-size control.
///
/// 6-stage, 5th-order propagator with embedded 4th-order error estimate.
/// The First-Same-As-Last (FSAL) property means only 6 function evaluations
/// are needed per accepted step (the 7th stage reuses the first of the next step).
///
/// Step is accepted when the mixed error norm satisfies
/// \f[
///   \left\|\frac{e_i}{\text{atol} + |y_i|\,\text{rtol}}\right\|_\infty \leq 1.
/// \f]
///
/// Step size is updated via PI control:
/// \f[
///   h_{\text{new}} = h \cdot \min\!\left(10,\,\max\!\left(0.1,\;0.9\,
///     \|\varepsilon\|^{-0.2}\right)\right).
/// \f]
///
/// @param rtol      Relative tolerance (default 1e-6)
/// @param atol      Absolute tolerance (default 1e-9)
/// @param h0        Initial step-size hint (default 1e-3)
/// @param max_steps Hard limit on accepted steps; sets converged=false if hit
ODEResult ode_rk45(ODERhsFn f, Vector y0, real t0, real t1,
                   real rtol = 1e-6, real atol = 1e-9,
                   real h0 = 1e-3, idx max_steps = 1000000);

/// @brief Velocity Verlet, 2nd-order symplectic, 1 force evaluation per step.
///
/// Kick-drift-kick (KDK) form with FSAL acceleration caching:
/// \f[
///   q_{n+1} = q_n + h\,v_n + \tfrac{h^2}{2}\,a_n,
/// \f]
/// \f[
///   v_{n+1} = v_n + \tfrac{h}{2}(a_n + a_{n+1}),
/// \f]
/// where \f$ a_n = \text{accel}(q_n) \f$.
///
/// Conserves a modified Hamiltonian exactly, giving \f$ O(h^2) \f$ bounded
/// energy oscillation rather than secular drift.
SymplecticResult ode_verlet(AccelFn accel, Vector q0, Vector v0,
                             real t0, real t1, real h);

/// @brief Yoshida 4th-order symplectic, 3 force evaluations per step.
///
/// Composes three leapfrog sub-steps using Yoshida (1990) coefficients:
/// \f[
///   w_1 = \frac{1}{2 - 2^{1/3}}, \qquad w_0 = 1 - 2w_1.
/// \f]
/// Position (drift) and velocity (kick) coefficients:
/// \f[
///   c_1 = c_4 = \tfrac{w_1}{2}, \quad
///   c_2 = c_3 = \tfrac{w_0 + w_1}{2}, \quad
///   d_1 = d_3 = w_1, \quad d_2 = w_0.
/// \f]
///
/// Preferred over Verlet when accuracy matters and force evaluation is cheap.
SymplecticResult ode_yoshida4(AccelFn accel, Vector q0, Vector v0,
                               real t0, real t1, real h);

/// @brief RK4 for second-order systems q'' = accel(q), Nystrom form.
///
/// q and v are kept separate (same call signature as ode_verlet / ode_yoshida4)
/// so switching integrators requires no restructuring.
///
/// @note Not symplectic: energy drifts O(h^4) per step.
///       Prefer ode_verlet or ode_yoshida4 for long Hamiltonian runs.
SymplecticResult ode_rk4_2nd(AccelFn accel, Vector q0, Vector v0,
                              real t0, real t1, real h);

} // namespace num
