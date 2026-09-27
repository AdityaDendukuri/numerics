/// @file stochastic/rng.hpp
/// @brief The library's random engines and seeding utilities.
#pragma once
#include <random>

namespace num {

/// @name Random engines
///
/// The library's default engines, named once here. They are plain aliases: `num::rng` is
/// `std::mt19937`.
/// @{

/// @brief The default 32-bit engine, used wherever a sampler takes a generator.
using rng = std::mt19937;

/// @brief The 64-bit engine, used by the randomized graph algorithms.
using rng64 = std::mt19937_64;

/// @}

namespace markov {

/// @brief Construct an RNG seeded from hardware entropy.
///
/// Equivalent to `RNG(std::random_device{}())`. Use this at simulation
/// startup for non-deterministic seeds.
///
/// @tparam RNG  Any standard-library-compatible random number engine.
///              Defaults to num::rng.
template <typename RNG = rng>
RNG make_seeded_rng() {
    std::random_device rd;
    return RNG(rd());
}

/// @brief Construct an RNG from a fixed seed (for reproducible runs).
template <typename RNG = rng>
RNG make_rng(typename RNG::result_type seed) {
    return RNG(seed);
}

} // namespace markov
} // namespace num
