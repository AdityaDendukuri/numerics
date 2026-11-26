/// @file stats/probability.hpp
/// @brief Probability-vector utilities.
#pragma once

#include "core/types.hpp"
#include <algorithm>
#include <span>

namespace num {

/// @brief Compute a weighted sum: \f$\sum_i w_i\, g(i)\f$.
template <typename Projection>
[[nodiscard]] real weighted_sum(view<const real> weights, Projection &&value) {
    real result = 0.0;
    for (idx index = 0; index < weights.size(); ++index) {
        result += weights[index] * value(index);
    }
    return result;
}

/// @brief Set negative entries to zero and scale to sum 1:
/// \f$p_i \leftarrow \max(p_i, 0) / \sum_j \max(p_j, 0)\f$.
/// Returns the mass before normalizing.
inline real clip_and_normalize_nonnegative(view<real> values) {
    real mass = 0.0;
    for (real &value : values) {
        value = std::max(0.0, value);
        mass += value;
    }
    if (mass > 0.0) {
        const real inverse = 1.0 / mass;
        for (real &value : values) {
            value *= inverse;
        }
    }
    return mass;
}

} // namespace num
