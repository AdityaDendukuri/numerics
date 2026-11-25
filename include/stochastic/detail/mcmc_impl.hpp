/// @file stochastic/detail/mcmc_impl.hpp
/// @brief Metropolis sweep implementations; included by `stochastic/mcmc.hpp`.
#pragma once
#include "stochastic/mcmc.hpp"
#include <cmath>
#include <random>

namespace num::markov {

/// @brief Metropolis sweep: `n_sites` updates, accepted with \f$\min(1, e^{-\beta \Delta E})\f$.
/// `delta_energy(i)` returns \f$\Delta E\f$; `apply(i)` makes the move.
template <energy_difference DeltaE, std::invocable<idx> Apply,
          std::uniform_random_bit_generator RNG>
metropolis_stats metropolis_sweep(idx n_sites, DeltaE delta_energy, Apply apply_flip, real beta,
                                  RNG &rng) {
    std::uniform_real_distribution<real> u01(0.0, 1.0);
    std::uniform_int_distribution<idx> site_dist(0, n_sites - 1);

    metropolis_stats stats;
    stats.total = n_sites;

    for (idx k = 0; k < n_sites; ++k) {
        idx i = site_dist(rng);
        real dE = delta_energy(i);
        real p = (dE <= 0.0) ? 1.0 : std::exp(-beta * dE);
        if (p >= 1.0 || u01(rng) < p) {
            apply_flip(i);
            ++stats.accepted;
        }
    }
    return stats;
}

/// @brief Metropolis sweep, precomputed: `prob(i)` gives \f$\min(1, e^{-\beta \Delta E})\f$.
template <typename ProbFn, std::invocable<idx> Apply, std::uniform_random_bit_generator RNG>
metropolis_stats metropolis_sweep_prob(idx n_sites, ProbFn acceptance_prob, Apply apply_flip,
                                       RNG &rng) {
    std::uniform_real_distribution<real> u01(0.0, 1.0);
    std::uniform_int_distribution<idx> site_dist(0, n_sites - 1);

    metropolis_stats stats;
    stats.total = n_sites;

    for (idx k = 0; k < n_sites; ++k) {
        idx i = site_dist(rng);
        real p = acceptance_prob(i);
        if (p >= 1.0 || u01(rng) < p) {
            apply_flip(i);
            ++stats.accepted;
        }
    }
    return stats;
}

/// @brief Umbrella-sampled Metropolis sweep: rolled back if the order parameter leaves the window.
/// `save()` and `restore()` bracket the sweep; `measure()` returns the order parameter.
template <energy_difference DeltaE, std::invocable<idx> Apply, typename Save, typename Restore,
          typename Measure, std::uniform_random_bit_generator RNG>
umbrella_stats umbrella_sweep(idx n_sites, DeltaE delta_energy, Apply apply_flip, Save save_state,
                              Restore restore_state, Measure measure_order, umbrella_window window,
                              real beta, RNG &rng) {
    save_state();
    metropolis_stats mc = metropolis_sweep(n_sites, delta_energy, apply_flip, beta, rng);
    idx op = measure_order();

    umbrella_stats stats;
    stats.mc = mc;
    stats.order_param = op;

    if (!window.contains(op)) {
        restore_state();
        stats.reverted = true;
        stats.order_param = measure_order();
    }
    return stats;
}

/// @brief Umbrella-sampled Metropolis sweep, precomputed acceptance.
template <typename ProbFn, std::invocable<idx> Apply, typename Save, typename Restore,
          typename Measure, std::uniform_random_bit_generator RNG>
umbrella_stats umbrella_sweep_prob(idx n_sites, ProbFn acceptance_prob, Apply apply_flip,
                                   Save save_state, Restore restore_state, Measure measure_order,
                                   umbrella_window window, RNG &rng) {
    save_state();
    metropolis_stats mc = metropolis_sweep_prob(n_sites, acceptance_prob, apply_flip, rng);
    idx op = measure_order();

    umbrella_stats stats;
    stats.mc = mc;
    stats.order_param = op;

    if (!window.contains(op)) {
        restore_state();
        stats.reverted = true;
        stats.order_param = measure_order();
    }
    return stats;
}

} // namespace num::markov
