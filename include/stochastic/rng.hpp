/// @file stochastic/rng.hpp
/// @brief Random engines and seeding.
#pragma once
#include <random>

namespace num {

/// @name Random engines
/// @{

/// @brief Default 32-bit random generator: `std::mt19937`.
using rng = std::mt19937;

/// @brief 64-bit random generator: `std::mt19937_64`.
using rng64 = std::mt19937_64;

/// @}

namespace markov {

/// @brief Construct a generator seeded from hardware entropy.
template <typename RNG = rng>
RNG make_seeded_rng() {
    std::random_device rd;
    return RNG(rd());
}

/// @brief Construct a generator from a fixed seed.
template <typename RNG = rng>
RNG make_rng(typename RNG::result_type seed) {
    return RNG(seed);
}

} // namespace markov
} // namespace num
