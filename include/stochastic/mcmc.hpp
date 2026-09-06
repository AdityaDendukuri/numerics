/// @file stochastic/mcmc.hpp
/// @brief Metropolis-Hastings sweep API.
#pragma once
#include "core/types.hpp"

namespace num::markov {

/// Accepted and attempted proposal counts for a Metropolis sweep.
struct metropolis_stats {
    idx accepted = 0;
    idx total = 0;
    /// Return accepted/total, or zero when no proposals were attempted.
    [[nodiscard]] real acceptance_rate() const {
        return total > 0 ? static_cast<real>(accepted) / total : 0.0;
    }
};

/// Metropolis statistics plus umbrella-window rollback state.
struct umbrella_stats {
    metropolis_stats mc;
    bool reverted = false;
    idx order_param = 0;
};

/// Inclusive interval for an umbrella-sampling order parameter.
struct umbrella_window {
    idx lo = 0;
    idx hi = 0;
    /// Test whether an order parameter lies in the window.
    [[nodiscard]] bool contains(idx v) const { return v >= lo && v <= hi; }
};

} // namespace num::markov

#include "stochastic/detail/mcmc_impl.hpp"
