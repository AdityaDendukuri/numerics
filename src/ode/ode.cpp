#include "ode/ode.hpp"
#include <algorithm>
#include <cmath>

namespace num {


static void axpy_vec(real alpha, const Vector& x, Vector& y) {
    for (idx i = 0; i < x.size(); ++i) y[i] += alpha * x[i];
}

static real eps_guard(real t1) { return 1e-14 * std::abs(t1); }


EulerSteps::EulerSteps(ODERhsFn f, Vector y0, ODEParams p)
    : f_(std::move(f)), y_(std::move(y0)), dydt_(y_.size()),
      t_(p.t0), t1_(p.tf), h_(p.h) {}

void EulerSteps::advance() {
    if (t_ >= t1_ - eps_guard(t1_)) { done_ = true; return; }
    real dt = std::min(h_, t1_ - t_);
    f_(t_, y_, dydt_);
    axpy_vec(dt, dydt_, y_);
    t_ += dt;
    ++steps_;
}

ODEResult EulerSteps::run() {
    while (!done_) advance();
    return {std::move(y_), t1_, steps_, true};
}


RK4Steps::RK4Steps(ODERhsFn f, Vector y0, ODEParams p)
    : f_(std::move(f)), y_(std::move(y0)),
      k1_(y_.size()), k2_(y_.size()), k3_(y_.size()),
      k4_(y_.size()), ytmp_(y_.size()),
      t_(p.t0), t1_(p.tf), h_(p.h) {}

void RK4Steps::advance() {
    if (t_ >= t1_ - eps_guard(t1_)) { done_ = true; return; }
    const idx n  = y_.size();
    const real dt = std::min(h_, t1_ - t_);

    f_(t_, y_, k1_);
    for (idx i = 0; i < n; ++i) ytmp_[i] = y_[i] + 0.5*dt*k1_[i];
    f_(t_ + 0.5*dt, ytmp_, k2_);
    for (idx i = 0; i < n; ++i) ytmp_[i] = y_[i] + 0.5*dt*k2_[i];
    f_(t_ + 0.5*dt, ytmp_, k3_);
    for (idx i = 0; i < n; ++i) ytmp_[i] = y_[i] + dt*k3_[i];
    f_(t_ + dt, ytmp_, k4_);
    for (idx i = 0; i < n; ++i)
        y_[i] += (dt/6.0) * (k1_[i] + 2*k2_[i] + 2*k3_[i] + k4_[i]);
    t_ += dt;
    ++steps_;
}

ODEResult RK4Steps::run() {
    while (!done_) advance();
    return {std::move(y_), t1_, steps_, true};
}

//
// Butcher tableau from Dormand & Prince, "A family of embedded Runge-Kutta
// formulae", J. Comput. Appl. Math. 6(1), 19-26 (1980).
// 5th-order propagation, 4th-order embedded error estimate, FSAL property.

static constexpr real rk45_a21 = 1.0/5.0;
static constexpr real rk45_a31 = 3.0/40.0,      rk45_a32 = 9.0/40.0;
static constexpr real rk45_a41 = 44.0/45.0,     rk45_a42 = -56.0/15.0,     rk45_a43 = 32.0/9.0;
static constexpr real rk45_a51 = 19372.0/6561.0, rk45_a52 = -25360.0/2187.0,
                      rk45_a53 = 64448.0/6561.0, rk45_a54 = -212.0/729.0;
static constexpr real rk45_a61 = 9017.0/3168.0,  rk45_a62 = -355.0/33.0,
                      rk45_a63 = 46732.0/5247.0,  rk45_a64 = 49.0/176.0,
                      rk45_a65 = -5103.0/18656.0;

static constexpr real rk45_b1 = 35.0/384.0,  rk45_b3 = 500.0/1113.0,
                      rk45_b4 = 125.0/192.0, rk45_b5 = -2187.0/6784.0, rk45_b6 = 11.0/84.0;

static constexpr real rk45_e1 =  71.0/57600.0,  rk45_e3 = -71.0/16695.0,
                      rk45_e4 =  71.0/1920.0,   rk45_e5 = -17253.0/339200.0,
                      rk45_e6 =  22.0/525.0,    rk45_e7 = -1.0/40.0;

RK45Steps::RK45Steps(ODERhsFn f, Vector y0, ODEParams p)
    : f_(std::move(f)), y_(std::move(y0)),
      k1_(y_.size()), k2_(y_.size()), k3_(y_.size()), k4_(y_.size()),
      k5_(y_.size()), k6_(y_.size()), k7_(y_.size()),
      ytmp_(y_.size()), err_(y_.size()),
      t_(p.t0), t1_(p.tf), h_(std::min(p.h, p.tf - p.t0)),
      rtol_(p.rtol), atol_(p.atol), max_steps_(p.max_steps)
{
    f_(t_, y_, k1_);  // prime k1 for FSAL
}

void RK45Steps::advance() {
    if (t_ >= t1_ - eps_guard(t1_)) { done_ = true; return; }
    if (steps_ >= max_steps_)       { done_ = true; converged_ = false; return; }

    const idx n = y_.size();

    for (;;) {
        h_ = std::min(h_, t1_ - t_);

        for (idx i = 0; i < n; ++i) ytmp_[i] = y_[i] + h_*rk45_a21*k1_[i];
        f_(t_ + h_/5.0, ytmp_, k2_);

        for (idx i = 0; i < n; ++i)
            ytmp_[i] = y_[i] + h_*(rk45_a31*k1_[i] + rk45_a32*k2_[i]);
        f_(t_ + 3*h_/10.0, ytmp_, k3_);

        for (idx i = 0; i < n; ++i)
            ytmp_[i] = y_[i] + h_*(rk45_a41*k1_[i] + rk45_a42*k2_[i] + rk45_a43*k3_[i]);
        f_(t_ + 4*h_/5.0, ytmp_, k4_);

        for (idx i = 0; i < n; ++i)
            ytmp_[i] = y_[i] + h_*(rk45_a51*k1_[i] + rk45_a52*k2_[i]
                                  + rk45_a53*k3_[i] + rk45_a54*k4_[i]);
        f_(t_ + 8*h_/9.0, ytmp_, k5_);

        for (idx i = 0; i < n; ++i)
            ytmp_[i] = y_[i] + h_*(rk45_a61*k1_[i] + rk45_a62*k2_[i] + rk45_a63*k3_[i]
                                  + rk45_a64*k4_[i] + rk45_a65*k5_[i]);
        f_(t_ + h_, ytmp_, k6_);

        for (idx i = 0; i < n; ++i)
            ytmp_[i] = y_[i] + h_*(rk45_b1*k1_[i] + rk45_b3*k3_[i] + rk45_b4*k4_[i]
                                  + rk45_b5*k5_[i] + rk45_b6*k6_[i]);
        f_(t_ + h_, ytmp_, k7_);

        for (idx i = 0; i < n; ++i)
            err_[i] = h_*(rk45_e1*k1_[i] + rk45_e3*k3_[i] + rk45_e4*k4_[i]
                         + rk45_e5*k5_[i] + rk45_e6*k6_[i] + rk45_e7*k7_[i]);

        real err_norm = 0;
        for (idx i = 0; i < n; ++i) {
            real sc = atol_ + rtol_ * std::max(std::abs(y_[i]), std::abs(ytmp_[i]));
            err_norm = std::max(err_norm, std::abs(err_[i] / sc));
        }

        real factor = 0.9 * std::pow(err_norm + 1e-10, -0.2);
        factor = std::max(real(0.1), std::min(real(10.0), factor));

        if (err_norm <= 1.0) {
            t_ += h_;
            y_  = ytmp_;
            k1_ = k7_;   // FSAL
            ++steps_;
            h_ *= factor;
            return;
        }
        h_ *= factor;
    }
}

ODEResult RK45Steps::run() {
    while (!done_) advance();
    return {std::move(y_), t_, steps_, converged_};
}


VerletSteps::VerletSteps(AccelFn accel, Vector q0, Vector v0, ODEParams p)
    : accel_(std::move(accel)),
      q_(std::move(q0)), v_(std::move(v0)),
      a_cur_(q_.size()), a_next_(q_.size()),
      t_(p.t0), t1_(p.tf), h_(p.h)
{
    accel_(q_, a_cur_);  // prime initial acceleration
}

void VerletSteps::advance() {
    if (t_ >= t1_ - eps_guard(t1_)) { done_ = true; return; }
    const idx  n  = q_.size();
    const real dt = std::min(h_, t1_ - t_);

    for (idx i = 0; i < n; ++i)
        q_[i] += dt*v_[i] + 0.5*dt*dt*a_cur_[i];

    accel_(q_, a_next_);

    for (idx i = 0; i < n; ++i)
        v_[i] += 0.5*dt*(a_cur_[i] + a_next_[i]);

    std::swap(a_cur_, a_next_);
    t_ += dt;
    ++steps_;
}

SymplecticResult VerletSteps::run() {
    while (!done_) advance();
    return {std::move(q_), std::move(v_), t_, steps_};
}


Yoshida4Steps::Yoshida4Steps(AccelFn accel, Vector q0, Vector v0, ODEParams p)
    : accel_(std::move(accel)),
      q_(std::move(q0)), v_(std::move(v0)), acc_(q_.size()),
      t_(p.t0), t1_(p.tf), h_(p.h) {}

void Yoshida4Steps::advance() {
    if (t_ >= t1_ - eps_guard(t1_)) { done_ = true; return; }
    static const real w1 = 1.0 / (2.0 - std::cbrt(2.0));
    static const real w0 = 1.0 - 2.0*w1;
    static const real c1 = w1*0.5;
    static const real c2 = (w0 + w1)*0.5;
    static const real d1 = w1;
    static const real d2 = w0;

    const idx  n  = q_.size();
    const real dt = std::min(h_, t1_ - t_);

    for (idx i = 0; i < n; ++i) q_[i] += c1*dt*v_[i];
    accel_(q_, acc_);
    for (idx i = 0; i < n; ++i) v_[i] += d1*dt*acc_[i];
    for (idx i = 0; i < n; ++i) q_[i] += c2*dt*v_[i];
    accel_(q_, acc_);
    for (idx i = 0; i < n; ++i) v_[i] += d2*dt*acc_[i];
    for (idx i = 0; i < n; ++i) q_[i] += c2*dt*v_[i];
    accel_(q_, acc_);
    for (idx i = 0; i < n; ++i) v_[i] += d1*dt*acc_[i];
    for (idx i = 0; i < n; ++i) q_[i] += c1*dt*v_[i];

    t_ += dt;
    ++steps_;
}

SymplecticResult Yoshida4Steps::run() {
    while (!done_) advance();
    return {std::move(q_), std::move(v_), t_, steps_};
}


RK4_2ndSteps::RK4_2ndSteps(AccelFn accel, Vector q0, Vector v0, ODEParams p)
    : accel_(std::move(accel)),
      q_(std::move(q0)), v_(std::move(v0)),
      a1_(q_.size()), a2_(q_.size()), a3_(q_.size()), a4_(q_.size()),
      qtmp_(q_.size()),
      t_(p.t0), t1_(p.tf), h_(p.h) {}

void RK4_2ndSteps::advance() {
    if (t_ >= t1_ - eps_guard(t1_)) { done_ = true; return; }
    const idx  n  = q_.size();
    const real dt = std::min(h_, t1_ - t_);

    accel_(q_, a1_);
    for (idx i = 0; i < n; ++i) qtmp_[i] = q_[i] + 0.5*dt*v_[i];
    accel_(qtmp_, a2_);
    for (idx i = 0; i < n; ++i) qtmp_[i] = q_[i] + 0.5*dt*(v_[i] + 0.5*dt*a1_[i]);
    accel_(qtmp_, a3_);
    for (idx i = 0; i < n; ++i) qtmp_[i] = q_[i] + dt*(v_[i] + 0.5*dt*a2_[i]);
    accel_(qtmp_, a4_);

    for (idx i = 0; i < n; ++i) {
        q_[i] += dt*v_[i] + (dt*dt/6.0)*(a1_[i] + a2_[i] + a3_[i]);
        v_[i] += (dt/6.0)*(a1_[i] + 2.0*a2_[i] + 2.0*a3_[i] + a4_[i]);
    }
    t_ += dt;
    ++steps_;
}

SymplecticResult RK4_2ndSteps::run() {
    while (!done_) advance();
    return {std::move(q_), std::move(v_), t_, steps_};
}

// Factory functions

EulerSteps    euler   (ODERhsFn f, Vector y0, ODEParams p) { return EulerSteps(std::move(f), std::move(y0), p); }
RK4Steps      rk4     (ODERhsFn f, Vector y0, ODEParams p) { return RK4Steps(std::move(f), std::move(y0), p); }
RK45Steps     rk45    (ODERhsFn f, Vector y0, ODEParams p) { return RK45Steps(std::move(f), std::move(y0), p); }

VerletSteps   verlet  (AccelFn a, Vector q0, Vector v0, ODEParams p) { return VerletSteps(std::move(a), std::move(q0), std::move(v0), p); }
Yoshida4Steps yoshida4(AccelFn a, Vector q0, Vector v0, ODEParams p) { return Yoshida4Steps(std::move(a), std::move(q0), std::move(v0), p); }
RK4_2ndSteps  rk4_2nd (AccelFn a, Vector q0, Vector v0, ODEParams p){ return RK4_2ndSteps(std::move(a), std::move(q0), std::move(v0), p); }

// High-level wrappers

ODEResult ode_euler  (ODERhsFn f, Vector y0, ODEParams p) { return euler(std::move(f), std::move(y0), p).run(); }
ODEResult ode_rk4    (ODERhsFn f, Vector y0, ODEParams p) { return rk4(std::move(f), std::move(y0), p).run(); }
ODEResult ode_rk45   (ODERhsFn f, Vector y0, ODEParams p) { return rk45(std::move(f), std::move(y0), p).run(); }
SymplecticResult ode_verlet  (AccelFn a, Vector q0, Vector v0, ODEParams p)  { return verlet(std::move(a), std::move(q0), std::move(v0), p).run(); }
SymplecticResult ode_yoshida4(AccelFn a, Vector q0, Vector v0, ODEParams p)  { return yoshida4(std::move(a), std::move(q0), std::move(v0), p).run(); }
SymplecticResult ode_rk4_2nd (AccelFn a, Vector q0, Vector v0, ODEParams p)  { return rk4_2nd(std::move(a), std::move(q0), std::move(v0), p).run(); }

} // namespace num
