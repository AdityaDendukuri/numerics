/// @file stochastic/markov.hpp
/// @brief Random engines, sampling and Markov chain Monte Carlo.
///
/// Provides template-based Metropolis-Hastings sweeps and umbrella sampling
/// with zero-overhead callable dispatch (no std::function in hot loops).
///
/// Modules:
///   mcmc.hpp  -- metropolis_sweep, metropolis_sweep_prob,
///                umbrella_sweep, umbrella_sweep_prob
///   rng.hpp   -- make_seeded_rng, make_rng
#pragma once
#include "stochastic/boltzmann_table.hpp"
#include "stochastic/categorical.hpp"
#include "stochastic/mcmc.hpp"
#include "stochastic/probe.hpp"
#include "stochastic/rng.hpp"
