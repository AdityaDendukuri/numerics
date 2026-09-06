/// @file solve/sample.hpp
/// @brief sample(model, sampler, rng): the stochastic-sampling verb (MCMC), the
/// counterpart to solve() for targets that are sampled rather than solved.
#pragma once

#include "solve/algorithms.hpp"
#include "stochastic/mcmc.hpp"
#include <functional>

namespace num {

/// @brief An MCMC target: site count, per-site acceptance/proposal, observable.
struct mcmc_model {
    std::function<double(int)> accept_prob; ///< acceptance probability at a site
    std::function<void(int)> propose;       ///< propose and apply a move at a site
    int n_sites;
    std::function<double()> measure; ///< observable estimator, sampled each sweep
};

/// @brief Run equilibration then measurement sweeps; return the mean observable.
template <is_mcmc_alg A, typename RNG>
double sample(const mcmc_model &model, const A &alg, RNG &rng) {
    for (int s = 0; s < alg.equilibration; ++s) {
        markov::metropolis_sweep_prob(model.n_sites, model.accept_prob, model.propose, rng);
    }
    double acc = 0.0;
    for (int s = 0; s < alg.measurements; ++s) {
        markov::metropolis_sweep_prob(model.n_sites, model.accept_prob, model.propose, rng);
        acc += model.measure();
    }
    return acc / alg.measurements;
}

} // namespace num
