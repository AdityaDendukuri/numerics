/// @file stochastic/concepts.hpp
/// @brief Contracts for random engines, samplers, and Markov chain proposals.
#pragma once

#include "algebra/concepts.hpp"
#include "core/types.hpp"
#include <concepts>

namespace num {

/// @brief Uniform random bit generator, as the standard library defines one.
///
/// Stated so that a routine taking an engine says which one it means. Any
/// `std::mt19937`, `std::mt19937_64`, or user engine satisfying the standard's
/// requirements qualifies.
template <class G>
concept random_engine = std::uniform_random_bit_generator<G>;

/// @brief Sampler drawing an index from a fixed distribution.
///
/// \f[ P(X = k) = \frac{w_k}{\sum_j w_j} \f]
///
/// The weights are non-negative and need not sum to one. Whether a particular set
/// satisfies that is a property of the values, checked by
/// `num::stochastic::debug::verify_weights`.
template <class S, class G>
concept categorical_sampling = random_engine<G> && requires(S &sampler, G &rng) {
    { sampler(rng) } -> std::convertible_to<idx>;
};

/// @brief Energy difference for a proposed move, \f$\Delta E\f$.
///
/// Metropolis accepts with probability \f$\min(1, e^{-\beta \Delta E})\f$, so the
/// callable reports the change rather than the total, which is what makes a sweep
/// cost \f$O(1)\f$ per site instead of \f$O(N)\f$.
template <class F, class T = real>
concept energy_difference = scalars::field<T> && requires(F &&delta_e, idx site) {
    { delta_e(site) } -> std::convertible_to<T>;
};

} // namespace num
