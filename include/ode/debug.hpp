/// @file ode/debug.hpp
/// @brief Runtime verification of the laws an integrator is supposed to obey.
///
/// The order of accuracy is measured from step halving, which catches a wrong Butcher
/// tableau. Symplecticity is checked as preservation of \f$\omega = dq \wedge dp\f$, which bounds
/// long-time energy drift.
#pragma once

#include "container/vector.hpp"
#include "core/debug.hpp"
#include "ode/types.hpp"
#include <cmath>
#include <source_location>
#include <string>

namespace num::ode::debug {

using num::debug::diagnostic_level;
using num::debug::get_level;
using num::debug::panic;

/// @brief Measure the observed order of accuracy of a one-step integrator.
///
/// Integrates with steps \f$h\f$ and \f$h/2\f$ against a reference at \f$h/16\f$. The error
/// ratio approaches \f$2^p\f$, so its \f$\log_2\f$ recovers \f$p\f$.
///
/// @param advance Callable `(real t0, real t1, real h, const vec &y0, vec &y1)`.
/// @param y0 Initial state.
/// @param t0 Start of the integration interval.
/// @param t1 End of the integration interval.
/// @param claimed_order The order the method advertises.
/// @param slack Tolerated shortfall in the measured order.
/// @param loc Call site reported in the diagnostic.
template <class Advance>
inline void verify_order_of_accuracy(Advance &&advance, const vec &y0, real t0, real t1,
                                     real claimed_order, real slack = 0.4,
                                     std::source_location loc = std::source_location::current()) {
    if constexpr (!num::debug::sampling_compiled_in) {
        return;
    }
    if (num::debug::get_level() != num::debug::diagnostic_level::full) {
        return;
    }
    const real span = t1 - t0;
    if (!(span > real(0)) || y0.size() == 0) {
        return;
    }

    const real h = span / real(8);
    vec coarse(y0.size()), fine(y0.size()), reference(y0.size());
    advance(t0, t1, h, y0, coarse);
    advance(t0, t1, h / real(2), y0, fine);
    advance(t0, t1, h / real(16), y0, reference);

    auto distance = [](const vec &a, const vec &b) {
        real sum = 0.0;
        for (idx i = 0; i < a.size(); ++i) {
            const real d = a[i] - b[i];
            sum += d * d;
        }
        return std::sqrt(sum);
    };

    const real error_coarse = distance(coarse, reference);
    const real error_fine = distance(fine, reference);

    // Both errors at round-off means the method is exact on this problem
    // (a linear system under a high-order method, say); there is no slope to fit.
    if (error_coarse < 1e-13 || error_fine < 1e-14) {
        return;
    }

    const real observed = std::log2(error_coarse / error_fine);
    if (observed < claimed_order - slack) {
        panic("PropertyError",
              "integrator order-of-accuracy check failed: measured order " +
                  std::to_string(observed) + " but the method claims " +
                  std::to_string(claimed_order) +
                  ". Halving the step did not reduce the error as the claimed order requires.",
              loc);
    }
}

/// @brief Verify that one step preserves the symplectic 2-form \f$\omega = dq \wedge dp\f$.
///
/// Two tangent vectors are propagated by finite differences, and their pairing
/// \f$\omega(u,v) = \delta q_u \cdot \delta p_v - \delta q_v \cdot \delta p_u\f$ must be unchanged.
///
/// @param step Callable `(const vec &q, const vec &p, real h, vec &q1, vec &p1)`.
/// @param q0 Position at which to test.
/// @param p0 Momentum at which to test.
/// @param h Step size.
/// @param tol Relative tolerance on the preserved form.
/// @param loc Call site reported in the diagnostic.
template <class Step>
inline void verify_symplectic_2form(Step &&step, const vec &q0, const vec &p0, real h,
                                    real tol = 1e-6,
                                    std::source_location loc = std::source_location::current()) {
    if constexpr (!num::debug::sampling_compiled_in) {
        return;
    }
    if (num::debug::get_level() != num::debug::diagnostic_level::full) {
        return;
    }
    const idx n = q0.size();
    if (n == 0 || p0.size() != n) {
        return;
    }

    const real eps = 1e-6;

    // Two independent tangent directions in phase space.
    auto flow = [&](const vec &q, const vec &p, vec &q1, vec &p1) { step(q, p, h, q1, p1); };

    vec base_q(n), base_p(n);
    flow(q0, p0, base_q, base_p);

    // Tangent 1: perturb q[0].  Tangent 2: perturb p[0].
    vec qa = q0, pa = p0, qb = q0, pb = p0;
    qa[0] += eps;
    pb[0] += eps;

    vec qa1(n), pa1(n), qb1(n), pb1(n);
    flow(qa, pa, qa1, pa1);
    flow(qb, pb, qb1, pb1);

    // Finite-difference tangents of the discrete flow.
    real omega_after = 0.0;
    for (idx i = 0; i < n; ++i) {
        const real dq_u = (qa1[i] - base_q[i]) / eps;
        const real dp_u = (pa1[i] - base_p[i]) / eps;
        const real dq_v = (qb1[i] - base_q[i]) / eps;
        const real dp_v = (pb1[i] - base_p[i]) / eps;
        omega_after += (dq_u * dp_v) - (dq_v * dp_u);
    }

    // Before the step the tangents are the unit vectors e_{q0} and e_{p0},
    // whose pairing is exactly 1.
    const real omega_before = 1.0;
    const real drift = std::abs(omega_after - omega_before);
    if (drift > tol) {
        panic("PropertyError",
              "symplectic 2-form is not preserved: omega changed by " + std::to_string(drift) +
                  " over one step (dq^dp = " + std::to_string(omega_after) +
                  ", expected 1). The integrator is NOT symplectic and will drift secularly.",
              loc);
    }
}

} // namespace num::ode::debug
