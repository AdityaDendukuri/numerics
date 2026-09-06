/// @file stochastic/boltzmann_table.hpp
/// @brief Boltzmann acceptance probability helpers for Metropolis MCMC.
#pragma once

#include "core/types.hpp"
#include <cmath>
#include <vector>

namespace num::markov {

/// Return min(1, exp(-beta*dE)) for a proposed energy change.
inline double boltzmann_accept(double dE, double beta) noexcept {
    return (dE <= 0.0) ? 1.0 : std::exp(-beta * dE);
}

/// Precompute Boltzmann acceptance probabilities for fixed energy changes.
inline array<double> make_boltzmann_table(const array<double> &dEs, double beta) {
    array<double> table(dEs.size());
    for (std::size_t i = 0; i < dEs.size(); ++i) {
        table[i] = boltzmann_accept(dEs[i], beta);
    }
    return table;
}

} // namespace num::markov
