/// @file stochastic/boltzmann_table.hpp
/// @brief Boltzmann acceptance probability helpers for Metropolis MCMC.
///
/// boltzmann_accept(dE, beta)    -- min(1, exp(-beta*dE)) inline, branch-free
/// exp make_boltzmann_table(dEs, beta) -- precompute table for a discrete dE
/// set
///
/// Typical usage (Ising model):
/// @code
///   // Replace: boltz[si][ni] = (dE <= 0) ? 1.0 : exp(-beta*dE)
///   boltz[si][ni] = num::markov::boltzmann_accept(dE, beta);
/// @endcode
#pragma once

#include <cmath>
#include <vector>

namespace num::markov {

/// Metropolis acceptance probability min(1, exp(-beta*dE)).
inline double boltzmann_accept(double dE, double beta) noexcept {
    return (dE <= 0.0) ? 1.0 : std::exp(-beta * dE);
}

/// Precompute a table of acceptance probabilities for a discrete set of dE
/// values.
inline std::vector<double> make_boltzmann_table(const std::vector<double>& dEs,
                                                double beta) {
    std::vector<double> table(dEs.size());
    for (std::size_t i = 0; i < dEs.size(); ++i)
        table[i] = boltzmann_accept(dEs[i], beta);
    return table;
}

} // namespace num::markov
