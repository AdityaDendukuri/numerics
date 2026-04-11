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
///
/// Parameters are passed via ODEParams using C++20 designated initializers:
/// \code
///   rk45(f, y0, {.tf = 50.0, .rtol = 1e-8, .atol = 1e-10})
/// \endcode
#pragma once

#include "core/types.hpp"
#include "core/vector.hpp"
#include "ode/implicit.hpp"
#include <functional>

namespace num {

using ODERhsFn   = std::function<void(real t, const Vector& y, Vector& dydt)>;
using AccelFn    = std::function<void(const Vector& q, Vector& acc)>;
using ObserverFn = std::function<void(real t, const Vector& y)>;
using SympObserverFn =
    std::function<void(real t, const Vector& q, const Vector& v)>;

struct Step {
    real   t = 0.0;
    Vector y;
};

struct SymplecticStep {
    real   t = 0.0;
    Vector q;
    Vector v;
};

struct ODEResult {
    Vector y;
    real   t         = 0.0;
    idx    steps     = 0;
    bool   converged = false;
};

struct SymplecticResult {
    Vector q;
    Vector v;
    real   t     = 0.0;
    idx    steps = 0;
};

/// Parameters for all ODE integrators. Set only the fields you need.
struct ODEParams {
    real t0   = 0.0;
    real tf   = 1.0;
    real h    = 1e-3; ///< step size (fixed-step) or initial hint (adaptive)
    real rtol = 1e-6; ///< relative tolerance (adaptive only)
    real atol = 1e-9; ///< absolute tolerance (adaptive only)
    idx  max_steps = 1000000; ///< step cap (adaptive only)
};

struct StepEnd {};

class EulerSteps {
    ODERhsFn f_ = nullptr;
    Vector   y_, dydt_;
    real     t_ = 0.0, t1_ = 0.0, h_ = 0.0;
    idx      steps_ = 0;
    bool     done_  = false;

    void advance();

  public:
    explicit EulerSteps(ODERhsFn f, Vector y0, ODEParams p);

    struct iterator {
        EulerSteps* owner_;
        Step operator*() const {
            return {owner_->t_, owner_->y_};
        }
        iterator& operator++() {
            owner_->advance();
            return *this;
        }
        bool operator!=(StepEnd) const {
            return !owner_->done_;
        }
        bool operator==(StepEnd) const {
            return owner_->done_;
        }
    };

    iterator begin() {
        advance();
        return {this};
    }
    StepEnd end() const {
        return {};
    }
    ODEResult run();
};

class RK4Steps {
    ODERhsFn f_ = nullptr;
    Vector   y_, k1_, k2_, k3_, k4_, ytmp_;
    real     t_ = 0.0, t1_ = 0.0, h_ = 0.0;
    idx      steps_ = 0;
    bool     done_  = false;

    void advance();

  public:
    explicit RK4Steps(ODERhsFn f, Vector y0, ODEParams p);

    struct iterator {
        RK4Steps* owner_;
        Step operator*() const {
            return {owner_->t_, owner_->y_};
        }
        iterator& operator++() {
            owner_->advance();
            return *this;
        }
        bool operator!=(StepEnd) const {
            return !owner_->done_;
        }
        bool operator==(StepEnd) const {
            return owner_->done_;
        }
    };

    iterator begin() {
        advance();
        return {this};
    }
    StepEnd end() const {
        return {};
    }
    ODEResult run();
};

class RK45Steps {
    ODERhsFn f_ = nullptr;
    Vector   y_, k1_, k2_, k3_, k4_, k5_, k6_, k7_, ytmp_, err_;
    real     t_ = 0.0, t1_ = 0.0, h_ = 0.0, rtol_ = 0.0, atol_ = 0.0;
    idx      steps_ = 0, max_steps_ = 0;
    bool     done_ = false, converged_ = true;

    void advance();

  public:
    explicit RK45Steps(ODERhsFn f, Vector y0, ODEParams p);

    struct iterator {
        RK45Steps* owner_;
        Step operator*() const {
            return {owner_->t_, owner_->y_};
        }
        iterator& operator++() {
            owner_->advance();
            return *this;
        }
        bool operator!=(StepEnd) const {
            return !owner_->done_;
        }
        bool operator==(StepEnd) const {
            return owner_->done_;
        }
    };

    iterator begin() {
        advance();
        return {this};
    }
    StepEnd end() const {
        return {};
    }
    ODEResult run();
};

class VerletSteps {
    AccelFn accel_ = nullptr;
    Vector  q_, v_, a_cur_, a_next_;
    real    t_ = 0.0, t1_ = 0.0, h_ = 0.0;
    idx     steps_ = 0;
    bool    done_  = false;

    void advance();

  public:
    explicit VerletSteps(AccelFn accel, Vector q0, Vector v0, ODEParams p);

    struct iterator {
        VerletSteps* owner_;
        SymplecticStep operator*() const {
            return {owner_->t_, owner_->q_, owner_->v_};
        }
        iterator& operator++() {
            owner_->advance();
            return *this;
        }
        bool operator!=(StepEnd) const {
            return !owner_->done_;
        }
        bool operator==(StepEnd) const {
            return owner_->done_;
        }
    };

    iterator begin() {
        advance();
        return {this};
    }
    StepEnd end() const {
        return {};
    }
    SymplecticResult run();
};

class Yoshida4Steps {
    AccelFn accel_ = nullptr;
    Vector  q_, v_, acc_;
    real    t_ = 0.0, t1_ = 0.0, h_ = 0.0;
    idx     steps_ = 0;
    bool    done_  = false;

    void advance();

  public:
    explicit Yoshida4Steps(AccelFn accel, Vector q0, Vector v0, ODEParams p);

    struct iterator {
        Yoshida4Steps* owner_;
        SymplecticStep operator*() const {
            return {owner_->t_, owner_->q_, owner_->v_};
        }
        iterator& operator++() {
            owner_->advance();
            return *this;
        }
        bool operator!=(StepEnd) const {
            return !owner_->done_;
        }
        bool operator==(StepEnd) const {
            return owner_->done_;
        }
    };

    iterator begin() {
        advance();
        return {this};
    }
    StepEnd end() const {
        return {};
    }
    SymplecticResult run();
};

class RK4_2ndSteps {
    AccelFn accel_ = nullptr;
    Vector  q_, v_, a1_, a2_, a3_, a4_, qtmp_;
    real    t_ = 0.0, t1_ = 0.0, h_ = 0.0;
    idx     steps_ = 0;
    bool    done_  = false;

    void advance();

  public:
    explicit RK4_2ndSteps(AccelFn accel, Vector q0, Vector v0, ODEParams p);

    struct iterator {
        RK4_2ndSteps* owner_;
        SymplecticStep operator*() const {
            return {owner_->t_, owner_->q_, owner_->v_};
        }
        iterator& operator++() {
            owner_->advance();
            return *this;
        }
        bool operator!=(StepEnd) const {
            return !owner_->done_;
        }
        bool operator==(StepEnd) const {
            return owner_->done_;
        }
    };

    iterator begin() {
        advance();
        return {this};
    }
    StepEnd end() const {
        return {};
    }
    SymplecticResult run();
};

// Lazy-range factories

EulerSteps euler(ODERhsFn f, Vector y0, ODEParams p = {});
RK4Steps rk4(ODERhsFn f, Vector y0, ODEParams p = {});
RK45Steps rk45(ODERhsFn f, Vector y0, ODEParams p = {});

VerletSteps verlet(AccelFn accel, Vector q0, Vector v0, ODEParams p = {});
Yoshida4Steps yoshida4(AccelFn accel, Vector q0, Vector v0, ODEParams p = {});
RK4_2ndSteps rk4_2nd(AccelFn accel, Vector q0, Vector v0, ODEParams p = {});

// High-level integrators — return final state only

/// @brief Forward Euler, 1st-order, fixed step.
ODEResult ode_euler(ODERhsFn   f,
                    Vector     y0,
                    ODEParams  p   = {},
                    ObserverFn obs = nullptr);

/// @brief Classic 4th-order Runge-Kutta, fixed step.
ODEResult ode_rk4(ODERhsFn   f,
                  Vector     y0,
                  ODEParams  p   = {},
                  ObserverFn obs = nullptr);

/// @brief Adaptive Dormand-Prince RK45 with FSAL and PI step-size control.
ODEResult ode_rk45(ODERhsFn   f,
                   Vector     y0,
                   ODEParams  p   = {},
                   ObserverFn obs = nullptr);

/// @brief Velocity Verlet, 2nd-order symplectic, 1 force evaluation per step.
SymplecticResult ode_verlet(AccelFn        accel,
                            Vector         q0,
                            Vector         v0,
                            ODEParams      p   = {},
                            SympObserverFn obs = nullptr);

/// @brief Yoshida 4th-order symplectic, 3 force evaluations per step.
SymplecticResult ode_yoshida4(AccelFn        accel,
                              Vector         q0,
                              Vector         v0,
                              ODEParams      p   = {},
                              SympObserverFn obs = nullptr);

/// @brief RK4 for second-order systems q'' = accel(q), Nystrom form.
/// @note Not symplectic. Prefer ode_verlet or ode_yoshida4 for long Hamiltonian
/// runs.
SymplecticResult ode_rk4_2nd(AccelFn        accel,
                             Vector         q0,
                             Vector         v0,
                             ODEParams      p   = {},
                             SympObserverFn obs = nullptr);

} // namespace num
