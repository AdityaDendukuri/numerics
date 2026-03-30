/// @file markov/detail/mcmc_impl.hpp
/// @brief Template implementations for markov/mcmc.hpp.
/// Included at the bottom of mcmc.hpp -- do not include directly.
#pragma once
#include "stochastic/mcmc.hpp"
#include <random>
#include <cmath>

namespace num::markov {

/// @brief Single Metropolis sweep: n_sites random single-site updates.
///
/// The acceptance probability is min(1, exp(-beta * dE)) computed from
/// the value returned by delta_energy(i).
///
/// @tparam DeltaE  Callable: idx -> real. Returns dE for proposing a flip at i.
/// @tparam Apply   Callable: idx -> void. Applies the flip at site i.
/// @tparam RNG     Random number generator (e.g., std::mt19937).
template<typename DeltaE, typename Apply, typename RNG>
MetropolisStats metropolis_sweep(
    idx    n_sites,
    DeltaE delta_energy,
    Apply  apply_flip,
    real   beta,
    RNG&   rng)
{
    std::uniform_real_distribution<real> u01(0.0, 1.0);
    std::uniform_int_distribution<idx>   site_dist(0, n_sites - 1);

    MetropolisStats stats;
    stats.total = n_sites;

    for (idx k = 0; k < n_sites; ++k) {
        idx  i  = site_dist(rng);
        real dE = delta_energy(i);
        real p  = (dE <= 0.0) ? 1.0 : std::exp(-beta * dE);
        if (p >= 1.0 || u01(rng) < p) {
            apply_flip(i);
            ++stats.accepted;
        }
    }
    return stats;
}

/// @brief Metropolis sweep with caller-supplied acceptance probabilities.
///
/// Use this variant when acceptance probabilities are precomputed (e.g. a
/// Boltzmann table for a discrete-dE system like the Ising model), avoiding
/// a runtime exp() call per proposed flip.
///
/// @tparam ProbFn  Callable: idx -> real. Returns min(1, exp(-beta*dE)) for site i.
/// @tparam Apply   Callable: idx -> void. Applies the flip at site i.
/// @tparam RNG     Random number generator.
template<typename ProbFn, typename Apply, typename RNG>
MetropolisStats metropolis_sweep_prob(
    idx    n_sites,
    ProbFn acceptance_prob,
    Apply  apply_flip,
    RNG&   rng)
{
    std::uniform_real_distribution<real> u01(0.0, 1.0);
    std::uniform_int_distribution<idx>   site_dist(0, n_sites - 1);

    MetropolisStats stats;
    stats.total = n_sites;

    for (idx k = 0; k < n_sites; ++k) {
        idx  i = site_dist(rng);
        real p = acceptance_prob(i);
        if (p >= 1.0 || u01(rng) < p) {
            apply_flip(i);
            ++stats.accepted;
        }
    }
    return stats;
}

/// @brief Umbrella-sampled Metropolis sweep (dE variant).
///
/// Performs a sweep, measures the order parameter, then restores the saved
/// state if the order parameter falls outside [window.lo, window.hi].
///
/// @tparam DeltaE   Callable: idx -> real
/// @tparam Apply    Callable: idx -> void
/// @tparam Save     Callable: () -> void. Saves system state before the sweep.
/// @tparam Restore  Callable: () -> void. Restores saved state.
/// @tparam Measure  Callable: () -> idx. Returns the order parameter.
/// @tparam RNG      Random number generator.
template<typename DeltaE, typename Apply,
         typename Save, typename Restore, typename Measure,
         typename RNG>
UmbrellaStats umbrella_sweep(
    idx            n_sites,
    DeltaE         delta_energy,
    Apply          apply_flip,
    Save           save_state,
    Restore        restore_state,
    Measure        measure_order,
    UmbrellaWindow window,
    real           beta,
    RNG&           rng)
{
    save_state();
    MetropolisStats mc = metropolis_sweep(n_sites, delta_energy, apply_flip, beta, rng);
    idx op = measure_order();

    UmbrellaStats stats;
    stats.mc          = mc;
    stats.order_param = op;

    if (!window.contains(op)) {
        restore_state();
        stats.reverted    = true;
        stats.order_param = measure_order();
    }
    return stats;
}

/// @brief Umbrella-sampled Metropolis sweep (precomputed probability variant).
///
/// Same as umbrella_sweep but uses caller-supplied acceptance probabilities.
///
/// @tparam ProbFn   Callable: idx -> real
/// @tparam Apply    Callable: idx -> void
/// @tparam Save     Callable: () -> void
/// @tparam Restore  Callable: () -> void
/// @tparam Measure  Callable: () -> idx
/// @tparam RNG      Random number generator.
template<typename ProbFn, typename Apply,
         typename Save, typename Restore, typename Measure,
         typename RNG>
UmbrellaStats umbrella_sweep_prob(
    idx            n_sites,
    ProbFn         acceptance_prob,
    Apply          apply_flip,
    Save           save_state,
    Restore        restore_state,
    Measure        measure_order,
    UmbrellaWindow window,
    RNG&           rng)
{
    save_state();
    MetropolisStats mc = metropolis_sweep_prob(n_sites, acceptance_prob, apply_flip, rng);
    idx op = measure_order();

    UmbrellaStats stats;
    stats.mc          = mc;
    stats.order_param = op;

    if (!window.contains(op)) {
        restore_state();
        stats.reverted    = true;
        stats.order_param = measure_order();
    }
    return stats;
}

} // namespace num::markov
