/// @file stochastic/multinomial.hpp
/// @brief Multinomial sampling.
#pragma once

#include "core/types.hpp"
#include "stochastic/concepts.hpp"
#include <algorithm>
#include <random>
#include <stdexcept>

namespace num {

/// @brief Draw multinomial counts \f$n_k\f$ with \f$P(k) \propto w_k\f$.
/// The counts sum to `draws`.
template <std::uniform_random_bit_generator RNG>
[[nodiscard]] array<idx> sample_multinomial(view<const real> weights, idx draws, RNG &random) {
    array<idx> count(weights.size(), 0);
    if (draws == 0) {
        return count;
    }
    if (weights.empty()) {
        throw std::invalid_argument("sample_multinomial: weights must not be empty");
    }

    real remaining_weight = 0.0;
    for (real weight : weights) {
        if (weight < 0.0) {
            throw std::invalid_argument("sample_multinomial: weights must be nonnegative");
        }
        remaining_weight += weight;
    }
    if (!(remaining_weight > 0.0)) {
        throw std::invalid_argument("sample_multinomial: at least one weight must be positive");
    }

    idx remaining = draws;
    for (idx k = 0; k + 1 < weights.size(); ++k) {
        if (remaining == 0) {
            break;
        }
        real probability = std::clamp(weights[k] / remaining_weight, real{0.0}, real{1.0});
        count[k] = std::binomial_distribution<idx>(remaining, probability)(random);
        remaining -= count[k];
        remaining_weight -= weights[k];
        if (!(remaining_weight > 0.0)) {
            break;
        }
    }
    count.back() += remaining;
    return count;
}

} // namespace num
