/// @file stochastic/boltzmann_table.hpp
/// @brief Metropolis acceptance probabilities.
#pragma once

#include "core/types.hpp"
#include <cmath>
#include <vector>

namespace num::markov {

/// @brief Compute the Metropolis acceptance probability: \f$\min(1, e^{-\beta \Delta E})\f$.
inline double boltzmann_accept(double dE, double beta) noexcept {
    return (dE <= 0.0) ? 1.0 : std::exp(-beta * dE);
}

/// @brief Tabulate the Metropolis acceptance probability for each \f$\Delta E_k\f$:
/// \f$\min(1, e^{-\beta \Delta E_k})\f$.
inline array<double> make_boltzmann_table(const array<double> &dEs, double beta) {
    array<double> table(dEs.size());
    for (std::size_t i = 0; i < dEs.size(); ++i) {
        table[i] = boltzmann_accept(dEs[i], beta);
    }
    return table;
}

} // namespace num::markov
