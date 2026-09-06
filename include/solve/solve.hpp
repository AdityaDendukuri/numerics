/// @file solve/solve.hpp
/// @brief Problem-level solve(): the uniform solve(problem, algorithm) -> result
/// verb over ode_problem and linear_problem. Repeated or warm-started linear solves
/// use init(problem, algorithm) + solve(cache). The low-level cg()/gmres()/...
/// kernels and ODE steppers remain the in-place primitives underneath. Stochastic
/// sampling (MCMC) lives in solve/sample.hpp as sample(model, sampler).
#pragma once

#include "linear/matrix_properties.hpp"
#include "linear/solvers/cg.hpp"
#include "linear/solvers/gmres.hpp"
#include "linear/solvers/minres.hpp"
#include "linear/solvers/pcg.hpp"
#include "linear/solvers/solver_result.hpp"
#include "ode/ode.hpp"
#include "operator/concepts.hpp"
#include "solve/algorithms.hpp"
#include "solve/problems.hpp"
#include <utility>

namespace num {

/// @brief Result of a linear solve: the solution vector plus convergence stats.
struct linear_solution {
    vec u;            ///< solution vector \f$\mathbf{u}\f$
    idx iterations = 0;  ///< iterations performed
    real residual = 0.0; ///< final residual norm \f$\|\mathbf{b} - A \mathbf{u}\|_2\f$
    bool converged = false;
};

/// @brief Solve an ODE initial value problem with adaptive Dormand-Prince (RK45) integration.
/// @tparam P ODE problem type satisfying `is_ode_problem`.
/// @param prob ODE problem instance holding RHS `f`, initial state `u0`, and time bounds `[t0, tf]`.
/// @param alg Algorithm configuration (`rk45_method{ .rtol=1e-6, .atol=1e-8 }`).
/// @param obs Optional step observer callback `void(real t, const vec& u)`.
/// @return `ode_result` containing final state `u`, time `t`, step counts, and convergence boolean.
template <is_ode_problem P>
ode_result solve(const P &prob, const rk45_method &alg, observer_fn obs = nullptr) {
    ode_params p{.t0 = prob.t0,
                .tf = prob.tf,
                .h = alg.h,
                .rtol = alg.rtol,
                .atol = alg.atol,
                .max_steps = alg.max_steps};
    return ode_rk45(prob.f, prob.u0, p, obs);
}

/// @brief Solve an ODE initial value problem with fixed-step classical 4th-order Runge-Kutta (RK4).
/// @tparam P ODE problem type satisfying `is_ode_problem`.
/// @param prob ODE problem instance.
/// @param alg Algorithm configuration (`rk4_method{ .h=0.01 }`).
/// @param obs Optional step observer callback.
/// @return `ode_result` containing final solution and stats.
template <is_ode_problem P>
ode_result solve(const P &prob, const rk4_method &alg, observer_fn obs = nullptr) {
    return ode_rk4(prob.f, prob.u0, {.t0 = prob.t0, .tf = prob.tf, .h = alg.h}, obs);
}

/// @brief Solve an ODE initial value problem with fixed-step forward Euler integration.
/// @tparam P ODE problem type satisfying `is_ode_problem`.
/// @param prob ODE problem instance.
/// @param alg Algorithm configuration (`euler_method{ .h=0.001 }`).
/// @param obs Optional step observer callback.
/// @return `ode_result` containing final solution and stats.
template <is_ode_problem P>
ode_result solve(const P &prob, const euler_method &alg, observer_fn obs = nullptr) {
    return ode_euler(prob.f, prob.u0, {.t0 = prob.t0, .tf = prob.tf, .h = alg.h}, obs);
}

// -- Linear systems: solve(linear_problem, CG/GMRES/MINRES/PCG) -> linear_solution --

namespace detail {

// The (operator x algorithm) dispatch, run in place into u (warm-startable).
// cg()/gmres()/minres()/pcg() are themselves overloaded on the operand type.

inline solver_result run(const linear::spd_mat<mat> &A, const vec &b, vec &u,
                        const cg_method &a) {
    return cg(A, b, u, a.tol, a.max_iter);
}

template <class Op>
requires spd_operator<Op, vec, vec> solver_result run(const Op &A, const vec &b, vec &u,
                                                          const cg_method &a) {
    return cg(A, b, u, a.tol, a.max_iter);
}

inline solver_result run(const mat &A, const vec &b, vec &u, const gmres_method &a) {
    return gmres(A, b, u, a.tol, a.max_iter, a.restart);
}

inline solver_result run(const spmat &A, const vec &b, vec &u, const gmres_method &a) {
    return gmres(A, b, u, a.tol, a.max_iter, a.restart);
}

template <class Op>
requires math::endomorphism_on<Op, vec> solver_result run(const Op &A, const vec &b, vec &u,
                                                           const gmres_method &a) {
    return gmres(A, b, u, a.tol, a.max_iter, a.restart);
}

template <class Op>
requires math::endomorphism_on<Op, vec> &&claims<Op, law::self_adjoint>
    solver_result run(const Op &A, const vec &b, vec &u, const minres_method &a) {
    return minres(A, b, u, a.tol, a.max_iter);
}

template <class Op, class M>
requires math::endomorphism_on<Op, vec> &&math::endomorphism_on<M, vec> &&
    claims<Op, law::spd> &&claims<M, law::spd>
        solver_result run(const Op &A, const vec &b, vec &u, const pcg_method<M> &a) {
    return pcg(A, a.preconditioner, b, u, a.tol, a.max_iter);
}

template <class Op, class M, class Subspace>
requires math::linear_subspace_of<Subspace, vec> &&math::endomorphism_on<Op, vec> &&
    math::endomorphism_on<M, vec> &&claims<Op, law::spd_on<Subspace>> &&
        claims<M, law::spd_on<Subspace>>
            solver_result run(const Op &A, const vec &b, vec &u, const pcg_on_method<M, Subspace> &a) {
    return pcg(A, a.preconditioner, b, u, a.subspace,
               pcg_options{.tolerance = a.tol, .max_iterations = a.max_iter});
}

} // namespace detail

/// @brief Reusable linear-solve cache (CommonSolve `init`/`solve!`): a view of
/// the problem plus the algorithm and the current iterate. Re-solving warm-starts
/// from cache.u. @note A and b are held by reference and must outlive the cache.
template <class Op, class Alg>
struct linear_cache {
    const Op &A;
    const vec &b;
    Alg alg;
    vec u; ///< warm-start on entry, solution on exit
};

/// @brief Build a solve cache; the iterate starts at zero.
template <class Op, class Alg>
linear_cache<Op, Alg> init(const linear_problem<Op> &prob, const Alg &alg) {
    return {prob.A, prob.b, alg, vec(prob.b.size(), real(0))};
}

/// @brief Build a solve cache seeded with an initial guess u0 (warm start).
template <class Op, class Alg>
linear_cache<Op, Alg> init(const linear_problem<Op> &prob, const Alg &alg, vec u0) {
    return {prob.A, prob.b, alg, std::move(u0)};
}

/// @brief Re-solve from a cache, warm-starting from its current iterate; cache.u
/// is updated in place (the C++ spelling of CommonSolve `solve!`).
template <class Op, class Alg>
linear_solution solve(linear_cache<Op, Alg> &cache) {
    const solver_result r = detail::run(cache.A, cache.b, cache.u, cache.alg);
    return {cache.u, r.iterations, r.residual, r.converged};
}

/// @brief One-shot linear solve: solve(problem, algorithm) == solve(init(...)).
template <class Op, class Alg>
linear_solution solve(const linear_problem<Op> &prob, const Alg &alg) {
    vec u(prob.b.size(), real(0));
    const solver_result r = detail::run(prob.A, prob.b, u, alg);
    return {std::move(u), r.iterations, r.residual, r.converged};
}

} // namespace num
