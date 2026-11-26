/// @file stochastic/concepts.hpp
/// @brief Concepts for random engines, samplers and Markov chain proposals.
#pragma once

#include "algebra/concepts.hpp"
#include "core/types.hpp"
#include <concepts>
#include <random>

namespace num {

/// @brief Sampler that draws index \f$k\f$ with probability \f$w_k / \sum_j w_j\f$, \f$w_k \ge
/// 0\f$.
template <class S, class G>
concept categorical_sampling = std::uniform_random_bit_generator<G> && requires(S &sampler, G &rng) {
    { sampler(rng) } -> std::convertible_to<idx>;
};

/// @brief Function that returns the energy change \f$\Delta E\f$ of a proposed move.
template <class F, class T = real>
concept energy_difference = field<T> && requires(F &&delta_e, idx site) {
    { delta_e(site) } -> std::convertible_to<T>;
};

} // namespace num
