/// @file ode/steps.hpp
/// @brief Lazy accepted-step ranges for first- and second-order ODE integrators.
#pragma once

#include "ode/types.hpp"

namespace num {

/// Lazy fixed-step forward Euler trajectory.
class EulerSteps {
    ODERhsFn f_ = nullptr;
    Vector y_, dydt_;
    real t_ = 0.0, t1_ = 0.0, h_ = 0.0;
    idx steps_ = 0;
    bool done_ = false;

    void advance();

  public:
    /// Initialize a trajectory at y0 over the interval in p.
    explicit EulerSteps(ODERhsFn f, Vector y0, ODEParams p);

    struct iterator {
        EulerSteps *owner_;
        Step operator*() const { return {owner_->t_, owner_->y_}; }
        iterator &operator++() {
            owner_->advance();
            return *this;
        }
        bool operator!=(StepEnd) const { return !owner_->done_; }
        bool operator==(StepEnd) const { return owner_->done_; }
    };

    iterator begin() {
        advance();
        return {this};
    }
    [[nodiscard]] StepEnd end() const { return {}; }
    /// Consume the remaining trajectory and return its final state.
    ODEResult run();
};

/// Lazy fixed-step classical fourth-order Runge-Kutta trajectory.
class RK4Steps {
    ODERhsFn f_ = nullptr;
    Vector y_, k1_, k2_, k3_, k4_, ytmp_;
    real t_ = 0.0, t1_ = 0.0, h_ = 0.0;
    idx steps_ = 0;
    bool done_ = false;

    void advance();

  public:
    /// Initialize a trajectory at y0 over the interval in p.
    explicit RK4Steps(ODERhsFn f, Vector y0, ODEParams p);

    struct iterator {
        RK4Steps *owner_;
        Step operator*() const { return {owner_->t_, owner_->y_}; }
        iterator &operator++() {
            owner_->advance();
            return *this;
        }
        bool operator!=(StepEnd) const { return !owner_->done_; }
        bool operator==(StepEnd) const { return owner_->done_; }
    };

    iterator begin() {
        advance();
        return {this};
    }
    [[nodiscard]] StepEnd end() const { return {}; }
    /// Consume the remaining trajectory and return its final state.
    ODEResult run();
};

/// Lazy adaptive Dormand-Prince trajectory with accepted-step iteration.
class RK45Steps {
    ODERhsFn f_ = nullptr;
    Vector y_, k1_, k2_, k3_, k4_, k5_, k6_, k7_, ytmp_, err_;
    real t_ = 0.0, t1_ = 0.0, h_ = 0.0, rtol_ = 0.0, atol_ = 0.0;
    idx steps_ = 0, max_steps_ = 0;
    bool done_ = false, converged_ = true;

    void advance();

  public:
    /// Initialize an adaptive trajectory at y0 over the interval in p.
    explicit RK45Steps(ODERhsFn f, Vector y0, ODEParams p);

    struct iterator {
        RK45Steps *owner_;
        Step operator*() const { return {owner_->t_, owner_->y_}; }
        iterator &operator++() {
            owner_->advance();
            return *this;
        }
        bool operator!=(StepEnd) const { return !owner_->done_; }
        bool operator==(StepEnd) const { return owner_->done_; }
    };

    iterator begin() {
        advance();
        return {this};
    }
    [[nodiscard]] StepEnd end() const { return {}; }
    /// Consume the remaining trajectory and return convergence metadata.
    ODEResult run();
};

/// Lazy velocity-Verlet trajectory for q''=a(q).
class VerletSteps {
    AccelFn accel_ = nullptr;
    Vector q_, v_, a_cur_, a_next_;
    real t_ = 0.0, t1_ = 0.0, h_ = 0.0;
    idx steps_ = 0;
    bool done_ = false;

    void advance();

  public:
    /// Initialize position and velocity over the interval in p.
    explicit VerletSteps(AccelFn accel, Vector q0, Vector v0, ODEParams p);

    struct iterator {
        VerletSteps *owner_;
        SymplecticStep operator*() const { return {owner_->t_, owner_->q_, owner_->v_}; }
        iterator &operator++() {
            owner_->advance();
            return *this;
        }
        bool operator!=(StepEnd) const { return !owner_->done_; }
        bool operator==(StepEnd) const { return owner_->done_; }
    };

    iterator begin() {
        advance();
        return {this};
    }
    [[nodiscard]] StepEnd end() const { return {}; }
    /// Consume the remaining trajectory and return its final state.
    SymplecticResult run();
};

/// Lazy fourth-order Yoshida symplectic trajectory for q''=a(q).
class Yoshida4Steps {
    AccelFn accel_ = nullptr;
    Vector q_, v_, acc_;
    real t_ = 0.0, t1_ = 0.0, h_ = 0.0;
    idx steps_ = 0;
    bool done_ = false;

    void advance();

  public:
    /// Initialize position and velocity over the interval in p.
    explicit Yoshida4Steps(AccelFn accel, Vector q0, Vector v0, ODEParams p);

    struct iterator {
        Yoshida4Steps *owner_;
        SymplecticStep operator*() const { return {owner_->t_, owner_->q_, owner_->v_}; }
        iterator &operator++() {
            owner_->advance();
            return *this;
        }
        bool operator!=(StepEnd) const { return !owner_->done_; }
        bool operator==(StepEnd) const { return owner_->done_; }
    };

    iterator begin() {
        advance();
        return {this};
    }
    [[nodiscard]] StepEnd end() const { return {}; }
    /// Consume the remaining trajectory and return its final state.
    SymplecticResult run();
};

/// Lazy non-symplectic fourth-order trajectory for q''=a(q).
class RK4_2ndSteps {
    AccelFn accel_ = nullptr;
    Vector q_, v_, a1_, a2_, a3_, a4_, qtmp_;
    real t_ = 0.0, t1_ = 0.0, h_ = 0.0;
    idx steps_ = 0;
    bool done_ = false;

    void advance();

  public:
    /// Initialize position and velocity over the interval in p.
    explicit RK4_2ndSteps(AccelFn accel, Vector q0, Vector v0, ODEParams p);

    struct iterator {
        RK4_2ndSteps *owner_;
        SymplecticStep operator*() const { return {owner_->t_, owner_->q_, owner_->v_}; }
        iterator &operator++() {
            owner_->advance();
            return *this;
        }
        bool operator!=(StepEnd) const { return !owner_->done_; }
        bool operator==(StepEnd) const { return owner_->done_; }
    };

    iterator begin() {
        advance();
        return {this};
    }
    [[nodiscard]] StepEnd end() const { return {}; }
    /// Consume the remaining trajectory and return its final state.
    SymplecticResult run();
};

} // namespace num
