/// @file stochastic/mcmc.hpp
/// @brief Metropolis–Hastings sweeps.
#pragma once
#include "core/types.hpp"

namespace num::markov {

/// @brief Counts from a sweep: proposals accepted and attempted.
struct metropolis_stats {
    idx accepted = 0;
    idx total = 0;
    /// @brief Return the fraction of proposals accepted; 0 when none were made.
    [[nodiscard]] real acceptance_rate() const {
        return total > 0 ? static_cast<real>(accepted) / total : 0.0;
    }
};

/// @brief Counts from a sweep confined to a window, including moves undone.
struct umbrella_stats {
    metropolis_stats mc;
    bool reverted = false;
    idx order_param = 0;
};

/// @brief Window \f$[\mathrm{lo}, \mathrm{hi}]\f$ to which the order parameter is confined.
struct umbrella_window {
    idx lo = 0;
    idx hi = 0;
    /// @brief Test whether \f$\mathrm{lo} \le v \le \mathrm{hi}\f$.
    [[nodiscard]] bool contains(idx v) const { return v >= lo && v <= hi; }
};

} // namespace num::markov

#include "stochastic/concepts.hpp"
#include "stochastic/detail/mcmc_impl.hpp"
