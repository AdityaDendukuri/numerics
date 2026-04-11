/// @file solve/solve.hpp
/// @brief Unified solve() dispatcher -- the single entry point for all solvers.
///
/// Dispatches on (problem type, algorithm type) via C++20 concepts, mirroring
/// Julia's SciML ecosystem where the algorithm is a swappable plug:
///
///   // Explicit ODE (Lorenz, nbody, quantum, ...)
///   auto sol = num::solve(ODEProblem{f, u0, t0, tf}, RK45{.rtol=1e-8}, obs);
///
///   // Implicit ODE (heat, diffusion, ...)
///   num::solve(u, BackwardEuler{.solver=cg, .dt=dt, .nstep=n});
///
///   // MCMC (Ising, spin glass, ...)
///   double m = num::solve(MCMCProblem{accept, flip, N}, Metropolis{}, measure, rng);
///
/// Problems carry the mathematics.  Algorithms carry the numerics.
/// Swapping the algorithm never touches the problem definition.
#pragma once

#include "solve/problems.hpp"
#include "solve/algorithms.hpp"
#include "ode/ode.hpp"
#include "ode/implicit.hpp"
#include "stochastic/mcmc.hpp"
#include <concepts>
#include <random>

namespace num {

// --- Concepts ----------------------------------------------------------------

template<typename P>
concept IsODEProblem = requires(const P& p, double t, const Vector& y, Vector& dy) {
    p.f(t, y, dy);
    { p.u0 } -> std::convertible_to<const Vector&>;
    { p.t0 } -> std::convertible_to<double>;
    { p.tf } -> std::convertible_to<double>;
};

template<typename A>
concept IsExplicitODEAlg =
    std::same_as<std::remove_cvref_t<A>, Euler> ||
    std::same_as<std::remove_cvref_t<A>, RK4>   ||
    std::same_as<std::remove_cvref_t<A>, RK45>;

template<typename A>
concept IsImplicitODEAlg = requires(const A& a) {
    { a.solver }  -> std::convertible_to<LinearSolver>;
    { a.dt }      -> std::convertible_to<double>;
    { a.nstep }   -> std::convertible_to<int>;
};

template<typename A>
concept IsMCMCAlg = std::same_as<std::remove_cvref_t<A>, Metropolis>;

// --- Explicit ODE ------------------------------------------------------------

template<IsODEProblem P>
ODEResult solve(const P& prob, const RK45& alg, ObserverFn obs = nullptr) {
    ODEParams p{.t0=prob.t0, .tf=prob.tf, .h=alg.h,
                .rtol=alg.rtol, .atol=alg.atol, .max_steps=alg.max_steps};
    return ode_rk45(prob.f, prob.u0, p, obs);
}

template<IsODEProblem P>
ODEResult solve(const P& prob, const RK4& alg, ObserverFn obs = nullptr) {
    return ode_rk4(prob.f, prob.u0, {.t0=prob.t0, .tf=prob.tf, .h=alg.h}, obs);
}

template<IsODEProblem P>
ODEResult solve(const P& prob, const Euler& alg, ObserverFn obs = nullptr) {
    return ode_euler(prob.f, prob.u0, {.t0=prob.t0, .tf=prob.tf, .h=alg.h}, obs);
}

// --- Implicit ODE ------------------------------------------------------------

template<VecField F>
void solve(F& u, const BackwardEuler& alg) {
    ode::advance(u, alg.solver, {alg.nstep, alg.dt});
}

template<VecField F, typename Observer>
void solve(F& u, const BackwardEuler& alg, Observer&& obs) {
    ode::advance(u, alg.solver, {alg.nstep, alg.dt}, std::forward<Observer>(obs));
}

// --- MCMC --------------------------------------------------------------------

/// Run equilibration + measurement sweeps; return the mean of measure() over
/// measurement sweeps.  The rng is passed by the caller so its state persists
/// across successive solve() calls (e.g. a temperature loop).
template<IsMCMCAlg A, typename MeasureFn, typename RNG>
double solve(const MCMCProblem& prob, const A& alg, MeasureFn&& measure, RNG& rng) {
    for (int s = 0; s < alg.equilibration; ++s)
        markov::metropolis_sweep_prob(prob.n_sites, prob.accept_prob, prob.propose, rng);
    double acc = 0.0;
    for (int s = 0; s < alg.measurements; ++s) {
        markov::metropolis_sweep_prob(prob.n_sites, prob.accept_prob, prob.propose, rng);
        acc += measure();
    }
    return acc / alg.measurements;
}

} // namespace num
