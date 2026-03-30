/// @file markov/markov.hpp
/// @brief Markov Chain Monte Carlo -- umbrella include.
///
/// Provides template-based Metropolis-Hastings sweeps and umbrella sampling
/// with zero-overhead callable dispatch (no std::function in hot loops).
///
/// Modules:
///   mcmc.hpp  -- metropolis_sweep, metropolis_sweep_prob,
///                umbrella_sweep, umbrella_sweep_prob
///   rng.hpp   -- make_seeded_rng, make_rng
#pragma once
#include "stochastic/mcmc.hpp"
#include "stochastic/rng.hpp"
#include "stochastic/boltzmann_table.hpp"
