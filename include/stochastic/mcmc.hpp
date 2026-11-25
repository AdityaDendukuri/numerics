/// @file stochastic/mcmc.hpp
/// @brief Metropolis–Hastings sweeps.
#pragma once
#include "core/types.hpp"

namespace num::markov {

/// @brief Sweep counts: accepted and attempted proposals.
struct metropolis_stats {
    idx accepted = 0;
    idx total = 0;
    /// @brief Acceptance rate: accepted / attempted, 0 when none.
    [[nodiscard]] real acceptance_rate() const {
        return total > 0 ? static_cast<real>(accepted) / total : 0.0;
    }
};

/// @brief Umbrella sweep counts, with rollbacks.
struct umbrella_stats {
    metropolis_stats mc;
    bool reverted = false;
    idx order_param = 0;
};

/// @brief Umbrella window: inclusive range \f$[\mathrm{lo}, \mathrm{hi}]\f$ of the order parameter.
struct umbrella_window {
    idx lo = 0;
    idx hi = 0;
    /// @brief Whether \f$\mathrm{lo} \le v \le \mathrm{hi}\f$.
    [[nodiscard]] bool contains(idx v) const { return v >= lo && v <= hi; }
};

} // namespace num::markov

#include "stochastic/concepts.hpp"
#include "stochastic/detail/mcmc_impl.hpp"
