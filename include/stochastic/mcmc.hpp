/// @file stochastic/mcmc.hpp
/// @brief Metropolis-Hastings sweep API.
#pragma once
#include "core/types.hpp"

namespace num::markov {

struct MetropolisStats {
  idx accepted = 0;
  idx total = 0;
  [[nodiscard]] real acceptance_rate() const {
    return total > 0 ? static_cast<real>(accepted) / total : 0.0;
  }
};

struct UmbrellaStats {
  MetropolisStats mc;
  bool reverted = false;
  idx order_param = 0;
};

struct UmbrellaWindow {
  idx lo = 0;
  idx hi = 0;
  [[nodiscard]] bool contains(idx v) const { return v >= lo && v <= hi; }
};

} // namespace num::markov

#include "stochastic/detail/mcmc_impl.hpp"
