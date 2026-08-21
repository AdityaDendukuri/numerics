/// @file stochastic/categorical.hpp
/// @brief Weighted categorical sampling.
#pragma once

#include "core/types.hpp"
#include <random>
#include <span>
#include <stdexcept>

namespace num {

template<typename RNG>
[[nodiscard]] idx sample_categorical(std::span<const real> weights, RNG& rng) {
  real total = 0.0;
  for (real weight : weights) {
    if (weight < 0.0) {
      throw std::invalid_argument("sample_categorical: weights must be nonnegative");
    }
    total += weight;
  }
  if (!(total > 0.0)) {
    throw std::invalid_argument(
      "sample_categorical: at least one weight must be positive");
  }
  std::uniform_real_distribution<real> uniform(0.0, total);
  const real target = uniform(rng);
  real cumulative = 0.0;
  for (idx index = 0; index < weights.size(); ++index) {
    cumulative += weights[index];
    if (target <= cumulative) {
      return index;
    }
  }
  return weights.size() - 1;
}

class CategoricalSampler {
public:
  explicit CategoricalSampler(std::span<const real> weights)
      : distribution_(weights.begin(), weights.end()) {}

  template<typename RNG>
  [[nodiscard]] idx operator()(RNG& rng) {
    return static_cast<idx>(distribution_(rng));
  }

private:
  std::discrete_distribution<std::size_t> distribution_;
};

} // namespace num
