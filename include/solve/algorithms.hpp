/// @file solve/algorithms.hpp
/// @brief Algorithm tags: carry the numerics, not the mathematics.
#pragma once

#include "core/policy.hpp"
#include "core/types.hpp"
#include <concepts>
#include <type_traits>

namespace num {

// -- ODE integrators --

struct Euler {
  double h = 1e-3;
};

struct RK4 {
  double h = 1e-3;
};

struct RK45 {
  double h = 1e-3;
  double rtol = 1e-6;
  double atol = 1e-9;
  idx max_steps = 1000000;
};

// -- Linear-system solvers --

struct CG {
  real tol = 1e-10;
  idx max_iter = 1000;
  Backend backend = default_backend;
};

struct GMRES {
  real tol = 1e-6;
  idx max_iter = 1000;
  idx restart = 30;
  Backend backend = default_backend;
};

struct MINRES {
  real tol = 1e-10;
  idx max_iter = 1000;
  Backend backend = default_backend;
};

template<class M>
struct PCG {
  const M& preconditioner;
  real tol = 1e-10;
  idx max_iter = 1000;
  Backend backend = default_backend;
};

template<class M>
PCG(const M&) -> PCG<M>;

// -- MCMC samplers --

struct Metropolis {
  int equilibration = 1000;
  int measurements = 500;
};

/// @brief An explicit ODE algorithm tag.
template<typename A>
concept IsExplicitODEAlg = std::same_as<std::remove_cvref_t<A>, Euler> ||
                           std::same_as<std::remove_cvref_t<A>, RK4> ||
                           std::same_as<std::remove_cvref_t<A>, RK45>;

/// @brief An MCMC algorithm tag.
template<typename A>
concept IsMCMCAlg = std::same_as<std::remove_cvref_t<A>, Metropolis>;


} // namespace num
