/// @file ode/steps.hpp
/// @brief Lazy accepted-step ranges for first- and second-order ODE integrators.
#pragma once

#include "container/vector.hpp"
#include "core/types.hpp"
#include "ode/types.hpp"
#include <algorithm>
#include <cmath>
#include <concepts>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>

namespace num {

namespace detail {

inline real eps_guard(real t1) {
    return 1e-14 * std::abs(t1);
}

/// @brief Reject integration parameters that cannot produce a trajectory.
///
/// Every stepper advances by `min(h, t1 - t)` and stops once `t` reaches `t1`.
/// Two parameter values break that silently rather than loudly: `h = 0` advances
/// time by nothing, so the loop never terminates; and `tf < t0` satisfies the
/// stop condition on the first call, so the integrator returns the initial
/// condition and reports success. Backward integration is not supported — say so
/// rather than appearing to do it.
inline void check_params(const ode_params &p, const char *method) {
    const std::string where = std::string(method) + ": ";
    if (!std::isfinite(p.t0) || !std::isfinite(p.tf)) {
        throw std::invalid_argument(where + "t0 and tf must be finite");
    }
    if (p.tf < p.t0) {
        throw std::invalid_argument(
            where + "tf must not precede t0; backward integration is not supported");
    }
    if (!(p.h > 0.0) || !std::isfinite(p.h)) {
        throw std::invalid_argument(where + "step size h must be finite and positive");
    }
    if (p.max_steps == 0) {
        throw std::invalid_argument(where + "max_steps must be at least 1");
    }
}

// Butcher tableau from Dormand & Prince (1980)
static constexpr real rk45_a21 = 1.0 / 5.0;
static constexpr real rk45_a31 = 3.0 / 40.0, rk45_a32 = 9.0 / 40.0;
static constexpr real rk45_a41 = 44.0 / 45.0, rk45_a42 = -56.0 / 15.0, rk45_a43 = 32.0 / 9.0;
static constexpr real rk45_a51 = 19372.0 / 6561.0, rk45_a52 = -25360.0 / 2187.0,
                      rk45_a53 = 64448.0 / 6561.0, rk45_a54 = -212.0 / 729.0;
static constexpr real rk45_a61 = 9017.0 / 3168.0, rk45_a62 = -355.0 / 33.0,
                      rk45_a63 = 46732.0 / 5247.0, rk45_a64 = 49.0 / 176.0,
                      rk45_a65 = -5103.0 / 18656.0;

static constexpr real rk45_b1 = 35.0 / 384.0, rk45_b3 = 500.0 / 1113.0, rk45_b4 = 125.0 / 192.0,
                      rk45_b5 = -2187.0 / 6784.0, rk45_b6 = 11.0 / 84.0;

static constexpr real rk45_e1 = 71.0 / 57600.0, rk45_e3 = -71.0 / 16695.0, rk45_e4 = 71.0 / 1920.0,
                      rk45_e5 = -17253.0 / 339200.0, rk45_e6 = 22.0 / 525.0, rk45_e7 = -1.0 / 40.0;

} // namespace detail

/// Lazy fixed-step forward Euler trajectory.
template <typename RHS = ode_rhs_fn, typename State = vec>
class basic_euler_steps {
    RHS f_{};
    State y_{}, dydt_{};
    real t_ = 0.0, t1_ = 0.0, h_ = 0.0;
    idx steps_ = 0, max_steps_ = 0;
    bool done_ = false, completed_ = false;

    void advance() {
        if (t_ >= t1_ - detail::eps_guard(t1_)) {
            done_ = true;
            completed_ = true;
            return;
        }
        if (steps_ >= max_steps_) {
            done_ = true; // work limit reached before t1: report it, do not spin
            return;
        }
        real dt = std::min(h_, t1_ - t_);
        f_(t_, y_, dydt_);
        for (idx i = 0; i < y_.size(); ++i) {
            y_[i] += dt * dydt_[i];
        }
        t_ += dt;
        ++steps_;
    }

  public:
    explicit basic_euler_steps(RHS f, State y0, ode_params p = {})
        : f_(std::move(f)), y_(std::move(y0)), dydt_(y_.size()), t_(p.t0), t1_(p.tf), h_(p.h) {
        detail::check_params(p, "ode_euler");
        max_steps_ = p.max_steps;
    }

    struct iterator {
        basic_euler_steps *owner_;
        ode_step operator*() const { return {owner_->t_, owner_->y_}; }
        iterator &operator++() {
            owner_->advance();
            return *this;
        }
        bool operator!=(step_end) const { return !owner_->done_; }
        bool operator==(step_end) const { return owner_->done_; }
    };

    iterator begin() {
        advance();
        return {this};
    }
    [[nodiscard]] step_end end() const { return {}; }

    ode_result run() {
        while (!done_) {
            advance();
        }
        return {std::move(y_), t_, steps_, completed_};
    }
};

using euler_steps = basic_euler_steps<ode_rhs_fn, vec>;

/// Lazy fixed-step classical fourth-order Runge-Kutta trajectory.
template <typename RHS = ode_rhs_fn, typename State = vec>
class basic_rk4_steps {
    RHS f_{};
    State y_{}, k1_{}, k2_{}, k3_{}, k4_{}, ytmp_{};
    real t_ = 0.0, t1_ = 0.0, h_ = 0.0;
    idx steps_ = 0, max_steps_ = 0;
    bool done_ = false, completed_ = false;

    void advance() {
        if (t_ >= t1_ - detail::eps_guard(t1_)) {
            done_ = true;
            completed_ = true;
            return;
        }
        if (steps_ >= max_steps_) {
            done_ = true; // work limit reached before t1: report it, do not spin
            return;
        }
        const idx n = y_.size();
        const real dt = std::min(h_, t1_ - t_);

        f_(t_, y_, k1_);
        for (idx i = 0; i < n; ++i) {
            ytmp_[i] = y_[i] + (0.5 * dt * k1_[i]);
        }
        f_(t_ + (0.5 * dt), ytmp_, k2_);
        for (idx i = 0; i < n; ++i) {
            ytmp_[i] = y_[i] + (0.5 * dt * k2_[i]);
        }
        f_(t_ + (0.5 * dt), ytmp_, k3_);
        for (idx i = 0; i < n; ++i) {
            ytmp_[i] = y_[i] + (dt * k3_[i]);
        }
        f_(t_ + dt, ytmp_, k4_);
        for (idx i = 0; i < n; ++i) {
            y_[i] += (dt / 6.0) * (k1_[i] + (2 * k2_[i]) + (2 * k3_[i]) + k4_[i]);
        }
        t_ += dt;
        ++steps_;
    }

  public:
    explicit basic_rk4_steps(RHS f, State y0, ode_params p = {})
        : f_(std::move(f)), y_(std::move(y0)), k1_(y_.size()), k2_(y_.size()), k3_(y_.size()),
          k4_(y_.size()), ytmp_(y_.size()), t_(p.t0), t1_(p.tf), h_(p.h) {
        detail::check_params(p, "ode_rk4");
        max_steps_ = p.max_steps;
    }

    struct iterator {
        basic_rk4_steps *owner_;
        ode_step operator*() const { return {owner_->t_, owner_->y_}; }
        iterator &operator++() {
            owner_->advance();
            return *this;
        }
        bool operator!=(step_end) const { return !owner_->done_; }
        bool operator==(step_end) const { return owner_->done_; }
    };

    iterator begin() {
        advance();
        return {this};
    }
    [[nodiscard]] step_end end() const { return {}; }

    ode_result run() {
        while (!done_) {
            advance();
        }
        return {std::move(y_), t_, steps_, completed_};
    }
};

using rk4_steps = basic_rk4_steps<ode_rhs_fn, vec>;

/// Lazy adaptive Dormand-Prince trajectory with accepted-step iteration.
template <typename RHS = ode_rhs_fn, typename State = vec>
class basic_rk45_steps {
    RHS f_{};
    State y_{}, k1_{}, k2_{}, k3_{}, k4_{}, k5_{}, k6_{}, k7_{}, ytmp_{}, err_{};
    real t_ = 0.0, t1_ = 0.0, h_ = 0.0, rtol_ = 0.0, atol_ = 0.0;
    idx steps_ = 0, max_steps_ = 0;
    bool done_ = false, converged_ = true;

    void advance() {
        if (t_ >= t1_ - detail::eps_guard(t1_)) {
            done_ = true;
            return;
        }
        if (steps_ >= max_steps_) {
            done_ = true;
            converged_ = false;
            return;
        }

        const idx n = y_.size();

        for (;;) {
            h_ = std::min(h_, t1_ - t_);

            for (idx i = 0; i < n; ++i) {
                ytmp_[i] = y_[i] + (h_ * detail::rk45_a21 * k1_[i]);
            }
            f_(t_ + (h_ / 5.0), ytmp_, k2_);

            for (idx i = 0; i < n; ++i) {
                ytmp_[i] =
                    y_[i] + (h_ * ((detail::rk45_a31 * k1_[i]) + (detail::rk45_a32 * k2_[i])));
            }
            f_(t_ + (3 * h_ / 10.0), ytmp_, k3_);

            for (idx i = 0; i < n; ++i) {
                ytmp_[i] =
                    y_[i] + (h_ * ((detail::rk45_a41 * k1_[i]) + (detail::rk45_a42 * k2_[i]) +
                                   (detail::rk45_a43 * k3_[i])));
            }
            f_(t_ + (4 * h_ / 5.0), ytmp_, k4_);

            for (idx i = 0; i < n; ++i) {
                ytmp_[i] =
                    y_[i] + (h_ * ((detail::rk45_a51 * k1_[i]) + (detail::rk45_a52 * k2_[i]) +
                                   (detail::rk45_a53 * k3_[i]) + (detail::rk45_a54 * k4_[i])));
            }
            f_(t_ + (8 * h_ / 9.0), ytmp_, k5_);

            for (idx i = 0; i < n; ++i) {
                ytmp_[i] =
                    y_[i] + (h_ * ((detail::rk45_a61 * k1_[i]) + (detail::rk45_a62 * k2_[i]) +
                                   (detail::rk45_a63 * k3_[i]) + (detail::rk45_a64 * k4_[i]) +
                                   (detail::rk45_a65 * k5_[i])));
            }
            f_(t_ + h_, ytmp_, k6_);

            for (idx i = 0; i < n; ++i) {
                ytmp_[i] = y_[i] + (h_ * ((detail::rk45_b1 * k1_[i]) + (detail::rk45_b3 * k3_[i]) +
                                          (detail::rk45_b4 * k4_[i]) + (detail::rk45_b5 * k5_[i]) +
                                          (detail::rk45_b6 * k6_[i])));
            }
            f_(t_ + h_, ytmp_, k7_);

            for (idx i = 0; i < n; ++i) {
                err_[i] = h_ * ((k1_[i] * detail::rk45_e1) + (detail::rk45_e3 * k3_[i]) +
                                (detail::rk45_e4 * k4_[i]) + (k5_[i] * detail::rk45_e5) +
                                (detail::rk45_e6 * k6_[i]) + (detail::rk45_e7 * k7_[i]));
            }

            real err_norm = 0;
            for (idx i = 0; i < n; ++i) {
                real sc = (rtol_ * std::max(std::abs(y_[i]), std::abs(ytmp_[i]))) + atol_;
                err_norm = std::max(err_norm, std::abs(err_[i] / sc));
            }

            real factor = 0.9 * std::pow(err_norm + 1e-10, -0.2);
            factor = std::max(real(0.1), std::min(real(10.0), factor));

            if (err_norm <= 1.0) {
                t_ += h_;
                y_ = ytmp_;
                k1_ = k7_; // FSAL
                ++steps_;
                h_ *= factor;
                return;
            }
            h_ *= factor;
        }
    }

  public:
    explicit basic_rk45_steps(RHS f, State y0, ode_params p = {})
        : f_(std::move(f)), y_(std::move(y0)), k1_(y_.size()), k2_(y_.size()), k3_(y_.size()),
          k4_(y_.size()), k5_(y_.size()), k6_(y_.size()), k7_(y_.size()), ytmp_(y_.size()),
          err_(y_.size()), t_(p.t0), t1_(p.tf), h_(std::min(p.h, p.tf - p.t0)), rtol_(p.rtol),
          atol_(p.atol), max_steps_(p.max_steps) {
        detail::check_params(p, "ode_rk45");
        f_(t_, y_, k1_); // prime k1 for FSAL
    }

    struct iterator {
        basic_rk45_steps *owner_;
        ode_step operator*() const { return {owner_->t_, owner_->y_}; }
        iterator &operator++() {
            owner_->advance();
            return *this;
        }
        bool operator!=(step_end) const { return !owner_->done_; }
        bool operator==(step_end) const { return owner_->done_; }
    };

    iterator begin() {
        advance();
        return {this};
    }
    [[nodiscard]] step_end end() const { return {}; }

    ode_result run() {
        while (!done_) {
            advance();
        }
        return {std::move(y_), t_, steps_, converged_};
    }
};

using rk45_steps = basic_rk45_steps<ode_rhs_fn, vec>;

/// Lazy velocity-Verlet trajectory for q''=a(q).
template <typename Accel = accel_fn, typename State = vec>
class basic_verlet_steps {
    Accel accel_{};
    State q_{}, v_{}, a_cur_{}, a_next_{};
    real t_ = 0.0, t1_ = 0.0, h_ = 0.0;
    idx steps_ = 0, max_steps_ = 0;
    bool done_ = false, completed_ = false;

    void advance() {
        if (t_ >= t1_ - detail::eps_guard(t1_)) {
            done_ = true;
            completed_ = true;
            return;
        }
        if (steps_ >= max_steps_) {
            done_ = true; // work limit reached before t1: report it, do not spin
            return;
        }
        const idx n = q_.size();
        const real dt = std::min(h_, t1_ - t_);

        for (idx i = 0; i < n; ++i) {
            q_[i] += (dt * v_[i]) + (0.5 * dt * dt * a_cur_[i]);
        }

        accel_(q_, a_next_);

        for (idx i = 0; i < n; ++i) {
            v_[i] += 0.5 * dt * (a_cur_[i] + a_next_[i]);
        }

        std::swap(a_cur_, a_next_);
        t_ += dt;
        ++steps_;
    }

  public:
    explicit basic_verlet_steps(Accel accel, State q0, State v0, ode_params p = {})
        : accel_(std::move(accel)), q_(std::move(q0)), v_(std::move(v0)), a_cur_(q_.size()),
          a_next_(q_.size()), t_(p.t0), t1_(p.tf), h_(p.h) {
        detail::check_params(p, "ode_verlet");
        max_steps_ = p.max_steps;
        accel_(q_, a_cur_); // prime initial acceleration
    }

    struct iterator {
        basic_verlet_steps *owner_;
        symplectic_step operator*() const { return {owner_->t_, owner_->q_, owner_->v_}; }
        iterator &operator++() {
            owner_->advance();
            return *this;
        }
        bool operator!=(step_end) const { return !owner_->done_; }
        bool operator==(step_end) const { return owner_->done_; }
    };

    iterator begin() {
        advance();
        return {this};
    }
    [[nodiscard]] step_end end() const { return {}; }

    symplectic_result run() {
        while (!done_) {
            advance();
        }
        return {std::move(q_), std::move(v_), t_, steps_};
    }
};

using verlet_steps = basic_verlet_steps<accel_fn, vec>;

/// Lazy fourth-order Yoshida symplectic trajectory for q''=a(q).
template <typename Accel = accel_fn, typename State = vec>
class basic_yoshida4_steps {
    Accel accel_{};
    State q_{}, v_{}, acc_{};
    real t_ = 0.0, t1_ = 0.0, h_ = 0.0;
    idx steps_ = 0, max_steps_ = 0;
    bool done_ = false, completed_ = false;

    void advance() {
        if (t_ >= t1_ - detail::eps_guard(t1_)) {
            done_ = true;
            completed_ = true;
            return;
        }
        if (steps_ >= max_steps_) {
            done_ = true; // work limit reached before t1: report it, do not spin
            return;
        }
        static const real w1 = 1.0 / (2.0 - std::cbrt(2.0));
        static const real w0 = 1.0 - (2.0 * w1);
        static const real c1 = w1 * 0.5;
        static const real c2 = (w0 + w1) * 0.5;
        static const real d1 = w1;
        static const real d2 = w0;

        const idx n = q_.size();
        const real dt = std::min(h_, t1_ - t_);

        for (idx i = 0; i < n; ++i) {
            q_[i] += c1 * dt * v_[i];
        }
        accel_(q_, acc_);
        for (idx i = 0; i < n; ++i) {
            v_[i] += d1 * dt * acc_[i];
        }

        for (idx i = 0; i < n; ++i) {
            q_[i] += c2 * dt * v_[i];
        }
        accel_(q_, acc_);
        for (idx i = 0; i < n; ++i) {
            v_[i] += d2 * dt * acc_[i];
        }

        for (idx i = 0; i < n; ++i) {
            q_[i] += c2 * dt * v_[i];
        }
        accel_(q_, acc_);
        for (idx i = 0; i < n; ++i) {
            v_[i] += d1 * dt * acc_[i];
        }

        for (idx i = 0; i < n; ++i) {
            q_[i] += c1 * dt * v_[i];
        }

        t_ += dt;
        ++steps_;
    }

  public:
    explicit basic_yoshida4_steps(Accel accel, State q0, State v0, ode_params p = {})
        : accel_(std::move(accel)), q_(std::move(q0)), v_(std::move(v0)), acc_(q_.size()), t_(p.t0),
          t1_(p.tf), h_(p.h) {
        detail::check_params(p, "ode_yoshida4");
        max_steps_ = p.max_steps;
    }

    struct iterator {
        basic_yoshida4_steps *owner_;
        symplectic_step operator*() const { return {owner_->t_, owner_->q_, owner_->v_}; }
        iterator &operator++() {
            owner_->advance();
            return *this;
        }
        bool operator!=(step_end) const { return !owner_->done_; }
        bool operator==(step_end) const { return owner_->done_; }
    };

    iterator begin() {
        advance();
        return {this};
    }
    [[nodiscard]] step_end end() const { return {}; }

    symplectic_result run() {
        while (!done_) {
            advance();
        }
        return {std::move(q_), std::move(v_), t_, steps_};
    }
};

using yoshida4_steps = basic_yoshida4_steps<accel_fn, vec>;

/// Lazy non-symplectic fourth-order trajectory for q''=a(q).
template <typename Accel = accel_fn, typename State = vec>
class basic_rk4_2nd_steps {
    Accel accel_{};
    State q_{}, v_{}, a1_{}, a2_{}, a3_{}, a4_{}, qtmp_{};
    real t_ = 0.0, t1_ = 0.0, h_ = 0.0;
    idx steps_ = 0, max_steps_ = 0;
    bool done_ = false, completed_ = false;

    void advance() {
        if (t_ >= t1_ - detail::eps_guard(t1_)) {
            done_ = true;
            completed_ = true;
            return;
        }
        if (steps_ >= max_steps_) {
            done_ = true; // work limit reached before t1: report it, do not spin
            return;
        }
        const idx n = q_.size();
        const real dt = std::min(h_, t1_ - t_);

        accel_(q_, a1_);

        for (idx i = 0; i < n; ++i) {
            qtmp_[i] = q_[i] + (0.5 * dt * v_[i]) + (0.125 * dt * dt * a1_[i]);
        }
        accel_(qtmp_, a2_);

        for (idx i = 0; i < n; ++i) {
            qtmp_[i] = q_[i] + (0.5 * dt * v_[i]) + (0.125 * dt * dt * a2_[i]);
        }
        accel_(qtmp_, a3_);

        for (idx i = 0; i < n; ++i) {
            qtmp_[i] = q_[i] + (dt * v_[i]) + (0.5 * dt * dt * a3_[i]);
        }
        accel_(qtmp_, a4_);

        for (idx i = 0; i < n; ++i) {
            q_[i] += (dt * v_[i]) + ((dt * dt / 6.0) * (a1_[i] + a2_[i] + a3_[i]));
            v_[i] += (dt / 6.0) * (a1_[i] + (2 * a2_[i]) + (2 * a3_[i]) + a4_[i]);
        }
        t_ += dt;
        ++steps_;
    }

  public:
    explicit basic_rk4_2nd_steps(Accel accel, State q0, State v0, ode_params p = {})
        : accel_(std::move(accel)), q_(std::move(q0)), v_(std::move(v0)), a1_(q_.size()),
          a2_(q_.size()), a3_(q_.size()), a4_(q_.size()), qtmp_(q_.size()), t_(p.t0), t1_(p.tf),
          h_(p.h) {
        detail::check_params(p, "ode_nystrom");
        max_steps_ = p.max_steps;
    }

    struct iterator {
        basic_rk4_2nd_steps *owner_;
        symplectic_step operator*() const { return {owner_->t_, owner_->q_, owner_->v_}; }
        iterator &operator++() {
            owner_->advance();
            return *this;
        }
        bool operator!=(step_end) const { return !owner_->done_; }
        bool operator==(step_end) const { return owner_->done_; }
    };

    iterator begin() {
        advance();
        return {this};
    }
    [[nodiscard]] step_end end() const { return {}; }

    symplectic_result run() {
        while (!done_) {
            advance();
        }
        return {std::move(q_), std::move(v_), t_, steps_};
    }
};

using rk4_2nd_steps = basic_rk4_2nd_steps<accel_fn, vec>;

} // namespace num
