/// @file solve/algorithms.hpp
/// @brief Algorithm tags: carry the numerics, not the mathematics.
#pragma once

#include "core/policy.hpp"
#include "core/types.hpp"
#include <concepts>
#include <type_traits>

namespace num {

// -- ODE integrators --

/// Fixed-step forward Euler configuration.
struct euler_method {
    double h = 1e-3;
};

/// Fixed-step classical fourth-order Runge-Kutta configuration.
struct rk4_method {
    double h = 1e-3;
};

/// Adaptive Dormand-Prince configuration.
struct rk45_method {
    double h = 1e-3;
    double rtol = 1e-6;
    double atol = 1e-9;
    idx max_steps = 1000000;
};

// -- Linear-system solvers --

/// Conjugate-gradient convergence options.
struct cg_method {
    real tol = 1e-10;
    idx max_iter = 1000;
};

/// Restarted GMRES convergence options.
struct gmres_method {
    real tol = 1e-6;
    idx max_iter = 1000;
    idx restart = 30;
};

/// MINRES convergence options.
struct minres_method {
    real tol = 1e-10;
    idx max_iter = 1000;
};

template <class M>
/// Preconditioned-CG configuration holding a non-owning preconditioner reference.
struct pcg_method {
    const M &preconditioner;
    real tol = 1e-10;
    idx max_iter = 1000;
};

template <class M>
pcg_method(const M &) -> pcg_method<M>;

template <class M, class Subspace>
/// PCG configuration for an operator and preconditioner certified on Subspace.
struct pcg_on_method {
    const M &preconditioner;
    Subspace subspace;
    real tol = 1e-10;
    idx max_iter = 1000;
};

template <class M, class Subspace>
pcg_on_method(const M &, Subspace) -> pcg_on_method<M, Subspace>;

// -- MCMC samplers --

/// Metropolis sampling burn-in and measurement counts.
struct metropolis_method {
    int equilibration = 1000;
    int measurements = 500;
};

/// @brief An explicit ODE algorithm tag.
template <typename A>
concept is_explicit_ode_alg =
    std::same_as<std::remove_cvref_t<A>, euler_method> || std::same_as<std::remove_cvref_t<A>, rk4_method> ||
    std::same_as<std::remove_cvref_t<A>, rk45_method>;

/// @brief An MCMC algorithm tag.
template <typename A>
concept is_mcmc_alg = std::same_as<std::remove_cvref_t<A>, metropolis_method>;

} // namespace num
