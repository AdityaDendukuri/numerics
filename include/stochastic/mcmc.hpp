/// @file markov/mcmc.hpp
/// @brief Metropolis-Hastings MCMC -- template-based, zero-overhead callable dispatch.
///
/// The sweep functions accept any callable for delta_energy / acceptance_prob
/// and apply_flip, so the compiler inlines them at the call site with zero
/// std::function overhead in the hot loop.
///
/// Two sweep variants are provided:
///   - metropolis_sweep:      caller supplies delta_energy (dE); exp(-beta*dE)
///                            is computed internally.
///   - metropolis_sweep_prob: caller supplies the acceptance probability
///                            directly (e.g. from a precomputed Boltzmann
///                            table), skipping the exp() call entirely.
///
/// Typical use (Ising with Boltzmann table):
/// @code
///   auto result = num::markov::metropolis_sweep_prob(
///       n_sites,
///       [&](idx i) { return boltz[spin_idx(i)][nbr_idx(i)]; },
///       [&](idx i) { spins[i] = -spins[i]; },
///       rng);
/// @endcode
///
/// Typical use (general system):
/// @code
///   auto result = num::markov::metropolis_sweep(
///       n_sites,
///       [&](idx i) { return sys.delta_energy(i); },
///       [&](idx i) { sys.flip(i); },
///       beta, rng);
/// @endcode
#pragma once
#include "core/types.hpp"

namespace num::markov {

/// @brief Statistics returned by a single Metropolis sweep.
struct MetropolisStats {
    idx  accepted = 0;  ///< Number of accepted proposals
    idx  total    = 0;  ///< Total proposals (= n_sites)
    real acceptance_rate() const {
        return total > 0 ? static_cast<real>(accepted) / total : 0.0;
    }
};

/// @brief Statistics returned by an umbrella sampling sweep.
struct UmbrellaStats {
    MetropolisStats mc;           ///< Metropolis sweep statistics
    bool reverted     = false;    ///< true if state was restored
    idx  order_param  = 0;        ///< Order parameter value after sweep
};

/// @brief Window constraint for umbrella sampling.
struct UmbrellaWindow {
    idx lo;  ///< Lower bound (inclusive)
    idx hi;  ///< Upper bound (inclusive)
    bool contains(idx v) const { return v >= lo && v <= hi; }
};

} // namespace num::markov

// Template definitions (must be visible at instantiation point).
#include "stochastic/detail/mcmc_impl.hpp"
