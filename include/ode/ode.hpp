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
/// All integrators accept an optional on_step callback invoked after each
/// accepted step, enabling trajectory recording without storing full history.
#pragma once

#include "core/types.hpp"
#include "core/vector.hpp"
#include <functional>

namespace num {

// Callable types

/// Right-hand side callable: fills dydt = f(t, y) in-place.
using ODERhsFn = std::function<void(real t, const Vector& y, Vector& dydt)>;

/// Called after each accepted step.
/// @param t  Time at end of step
/// @param y  State at end of step
using StepCallback = std::function<void(real t, const Vector& y)>;

/// Acceleration function for symplectic integrators.
/// Fills acc = -grad(V(q)) / m (generalised force per unit mass) from positions q.
using AccelFn = std::function<void(const Vector& q, Vector& acc)>;

/// Per-step callback for symplectic integrators.
/// @param t  Time at end of step
/// @param q  Generalised positions
/// @param v  Generalised velocities
using SymplecticCallback = std::function<void(real t, const Vector& q, const Vector& v)>;

// Result types

/// Result returned by general ODE integrators.
struct ODEResult {
    Vector y;          ///< Final state vector
    real   t;          ///< Final time
    idx    steps;      ///< Number of accepted steps taken
    bool   converged;  ///< false only if rk45 hit max_steps
};

/// Result returned by symplectic integrators.
struct SymplecticResult {
    Vector q;     ///< Final generalised positions
    Vector v;     ///< Final generalised velocities
    real   t;     ///< Final time
    idx    steps; ///< Number of steps taken
};

// General ODE solvers

/// @brief Forward Euler, 1st-order, fixed step.
///
/// \f[ y_{n+1} = y_n + h\,f(t_n,\,y_n) \f]
///
/// Local truncation error \f$ O(h^2) \f$; global error \f$ O(h) \f$.
ODEResult ode_euler(ODERhsFn f, Vector y0, real t0, real t1, real h,
                    StepCallback on_step = nullptr);

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
ODEResult ode_rk4(ODERhsFn f, Vector y0, real t0, real t1, real h,
                  StepCallback on_step = nullptr);

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
/// @param on_step   Optional callback after each accepted step
ODEResult ode_rk45(ODERhsFn f, Vector y0, real t0, real t1,
                   real rtol = 1e-6, real atol = 1e-9,
                   real h0 = 1e-3, idx max_steps = 1000000,
                   StepCallback on_step = nullptr);

// Symplectic integrators
//
// For separable Hamiltonians H(q,p) = T(p) + V(q) only.
// Symplectic (volume-preserving in phase space) and time-reversible;
// energy error is bounded rather than growing over exponentially long runs.
// RK4 has secular O(h^4) energy drift per step by contrast.
//
// Both methods take q and v separately (not concatenated) and an AccelFn.

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
                             real t0, real t1, real h,
                             SymplecticCallback on_step = nullptr);

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
/// One step executes the drift-kick sequence:
/// \f[
///   q \mathrel{+}= c_1 h v \to
///   v \mathrel{+}= d_1 h\,a(q) \to
///   q \mathrel{+}= c_2 h v \to
///   v \mathrel{+}= d_2 h\,a(q) \to
///   q \mathrel{+}= c_3 h v \to
///   v \mathrel{+}= d_3 h\,a(q) \to
///   q \mathrel{+}= c_4 h v.
/// \f]
///
/// Preferred over Verlet when accuracy matters and force evaluation is cheap.
SymplecticResult ode_yoshida4(AccelFn accel, Vector q0, Vector v0,
                               real t0, real t1, real h,
                               SymplecticCallback on_step = nullptr);

/// @brief RK4 for second-order systems q'' = accel(q), Nystrom form.
///
/// q and v are kept separate (same call signature as ode_verlet / ode_yoshida4)
/// so switching integrators requires no restructuring.
///
/// Stage accelerations evaluated at intermediate positions:
/// \f[
///   a_1 = \text{accel}(q), \quad
///   a_2 = \text{accel}\!\left(q + \tfrac{h}{2}v\right),
/// \f]
/// \f[
///   a_3 = \text{accel}\!\left(q + \tfrac{h}{2}\!\left(v + \tfrac{h}{2}a_1\right)\right), \quad
///   a_4 = \text{accel}\!\left(q + h\!\left(v + \tfrac{h}{2}a_2\right)\right).
/// \f]
///
/// Update:
/// \f[
///   q_{n+1} = q_n + h\,v_n + \tfrac{h^2}{6}(a_1 + a_2 + a_3),
/// \f]
/// \f[
///   v_{n+1} = v_n + \tfrac{h}{6}(a_1 + 2a_2 + 2a_3 + a_4).
/// \f]
///
/// @note Not symplectic: energy drifts O(h^4) per step.
///       Prefer ode_verlet or ode_yoshida4 for long Hamiltonian runs.
SymplecticResult ode_rk4_2nd(AccelFn accel, Vector q0, Vector v0,
                              real t0, real t1, real h,
                              SymplecticCallback on_step = nullptr);

} // namespace num
