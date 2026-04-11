/// @file markov/rng.hpp
/// @brief RNG seeding utilities for MCMC simulations.
#pragma once
#include <random>

namespace num::markov {

/// @brief Construct an RNG seeded from hardware entropy.
///
/// Equivalent to `RNG(std::random_device{}())`. Use this at simulation
/// startup for non-deterministic seeds.
///
/// @tparam RNG  Any standard-library-compatible random number engine.
///              Defaults to std::mt19937.
template<typename RNG = std::mt19937>
RNG make_seeded_rng() {
    std::random_device rd;
    return RNG(rd());
}

/// @brief Construct an RNG from a fixed seed (for reproducible runs).
template<typename RNG = std::mt19937>
RNG make_rng(typename RNG::result_type seed) {
    return RNG(seed);
}

} // namespace num::markov
