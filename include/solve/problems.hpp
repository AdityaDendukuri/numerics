/// @file solve/problems.hpp
/// @brief Problem types: carry the mathematics, not the numerics.
///
/// ODEProblem    -- du/dt = f(t, u),  u0,  time span [t0, tf]
/// MCMCProblem   -- Markov chain over n_sites with accept/propose callables
///
/// The algorithm (how to solve) is always separate; pass it to num::solve().
#pragma once

#include "core/vector.hpp"
#include "ode/ode.hpp"
#include <functional>
#include <random>

namespace num {

/// Explicit ODE:  du/dt = f(t, u)
struct ODEProblem {
    ODERhsFn f;
    Vector   u0;
    double   t0 = 0.0;
    double   tf = 1.0;
};

/// MCMC sampling problem over n_sites.
/// accept_prob(i) returns the Metropolis acceptance probability for site i.
/// propose(i)     applies the proposed move at site i.
/// The spin state is implicit in the lambda captures -- the caller owns it.
struct MCMCProblem {
    std::function<double(int)> accept_prob;
    std::function<void(int)>   propose;
    int                        n_sites;
};

} // namespace num
