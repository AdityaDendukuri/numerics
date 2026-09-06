/// @file stats/probability.hpp
/// @brief Probability-vector normalization utilities.
#pragma once

#include "core/types.hpp"
#include <algorithm>
#include <span>

namespace num {

/// Sum weights times values produced by an index projection.
template <typename Projection>
[[nodiscard]] real weighted_sum(view<const real> weights, Projection &&value) {
    real result = 0.0;
    for (idx index = 0; index < weights.size(); ++index) {
        result += weights[index] * value(index);
    }
    return result;
}

/// Clip negative entries to zero and normalize the remaining mass.
/// Returns the mass after clipping and before normalization.
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
