/// @file solve/solve.hpp
/// @brief Problem-level solve(): the uniform solve(problem, algorithm) -> result
/// verb over ODEProblem and LinearProblem. Repeated or warm-started linear solves
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
struct LinearSolution {
    Vector u;            ///< solution vector \f$\mathbf{u}\f$
    idx iterations = 0;  ///< iterations performed
    real residual = 0.0; ///< final residual norm \f$\|\mathbf{b} - A \mathbf{u}\|_2\f$
    bool converged = false;
};

/// @brief Solve an ODE initial value problem with adaptive Dormand-Prince (RK45) integration.
/// @tparam P ODE problem type satisfying `IsODEProblem`.
/// @param prob ODE problem instance holding RHS `f`, initial state `u0`, and time bounds `[t0, tf]`.
/// @param alg Algorithm configuration (`RK45{ .rtol=1e-6, .atol=1e-8 }`).
/// @param obs Optional step observer callback `void(real t, const Vector& u)`.
/// @return `ODEResult` containing final state `u`, time `t`, step counts, and convergence boolean.
template <IsODEProblem P>
ODEResult solve(const P &prob, const RK45 &alg, ObserverFn obs = nullptr) {
    ODEParams p{.t0 = prob.t0,
                .tf = prob.tf,
                .h = alg.h,
                .rtol = alg.rtol,
                .atol = alg.atol,
                .max_steps = alg.max_steps};
    return ode_rk45(prob.f, prob.u0, p, obs);
}

/// @brief Solve an ODE initial value problem with fixed-step classical 4th-order Runge-Kutta (RK4).
/// @tparam P ODE problem type satisfying `IsODEProblem`.
/// @param prob ODE problem instance.
/// @param alg Algorithm configuration (`RK4{ .h=0.01 }`).
/// @param obs Optional step observer callback.
/// @return `ODEResult` containing final solution and stats.
template <IsODEProblem P>
ODEResult solve(const P &prob, const RK4 &alg, ObserverFn obs = nullptr) {
    return ode_rk4(prob.f, prob.u0, {.t0 = prob.t0, .tf = prob.tf, .h = alg.h}, obs);
}

/// @brief Solve an ODE initial value problem with fixed-step forward Euler integration.
/// @tparam P ODE problem type satisfying `IsODEProblem`.
/// @param prob ODE problem instance.
/// @param alg Algorithm configuration (`Euler{ .h=0.001 }`).
/// @param obs Optional step observer callback.
/// @return `ODEResult` containing final solution and stats.
template <IsODEProblem P>
ODEResult solve(const P &prob, const Euler &alg, ObserverFn obs = nullptr) {
    return ode_euler(prob.f, prob.u0, {.t0 = prob.t0, .tf = prob.tf, .h = alg.h}, obs);
}

// -- Linear systems: solve(LinearProblem, CG/GMRES/MINRES/PCG) -> LinearSolution --

namespace detail {

// The (operator x algorithm) dispatch, run in place into u (warm-startable).
// cg()/gmres()/minres()/pcg() are themselves overloaded on the operand type.

inline SolverResult run(const linear::SPDMatrix<Matrix> &A, const Vector &b, Vector &u,
                        const CG &a) {
    return cg(A, b, u, a.tol, a.max_iter, a.backend);
}

template <class Op>
requires SPDOperator<Op, Vector, Vector> SolverResult run(const Op &A, const Vector &b, Vector &u,
                                                          const CG &a) {
    return cg(A, b, u, a.tol, a.max_iter, a.backend);
}

inline SolverResult run(const Matrix &A, const Vector &b, Vector &u, const GMRES &a) {
    return gmres(A, b, u, a.tol, a.max_iter, a.restart, a.backend);
}

inline SolverResult run(const SparseMatrix &A, const Vector &b, Vector &u, const GMRES &a) {
    return gmres(A, b, u, a.tol, a.max_iter, a.restart);
}

template <class Op>
requires math::EndomorphismOn<Op, Vector> SolverResult run(const Op &A, const Vector &b, Vector &u,
                                                           const GMRES &a) {
    return gmres(A, b, u, a.tol, a.max_iter, a.restart);
}

template <class Op>
requires math::EndomorphismOn<Op, Vector> &&math::Carries<Op, axiom::self_adjoint>
    SolverResult run(const Op &A, const Vector &b, Vector &u, const MINRES &a) {
    return minres(A, b, u, a.tol, a.max_iter, a.backend);
}

template <class Op, class M>
requires math::EndomorphismOn<Op, Vector> &&math::EndomorphismOn<M, Vector> &&
    math::Carries<Op, axiom::positive_definite> &&math::Carries<M, axiom::positive_definite>
        SolverResult run(const Op &A, const Vector &b, Vector &u, const PCG<M> &a) {
    return pcg(A, a.preconditioner, b, u, a.tol, a.max_iter, a.backend);
}

template <class Op, class M, class Subspace>
requires math::LinearSubspaceOf<Subspace, Vector> &&math::EndomorphismOn<Op, Vector> &&
    math::EndomorphismOn<M, Vector> &&math::Carries<Op, axiom::positive_definite_on<Subspace>> &&
        math::Carries<M, axiom::positive_definite_on<Subspace>>
            SolverResult run(const Op &A, const Vector &b, Vector &u, const PCGOn<M, Subspace> &a) {
    return pcg(A, a.preconditioner, b, u, a.subspace,
               PCGOptions{.tolerance = a.tol, .max_iterations = a.max_iter});
}

} // namespace detail

/// @brief Reusable linear-solve cache (CommonSolve `init`/`solve!`): a view of
/// the problem plus the algorithm and the current iterate. Re-solving warm-starts
/// from cache.u. @note A and b are held by reference and must outlive the cache.
template <class Op, class Alg>
struct LinearCache {
    const Op &A;
    const Vector &b;
    Alg alg;
    Vector u; ///< warm-start on entry, solution on exit
};

/// @brief Build a solve cache; the iterate starts at zero.
template <class Op, class Alg>
LinearCache<Op, Alg> init(const LinearProblem<Op> &prob, const Alg &alg) {
    return {prob.A, prob.b, alg, Vector(prob.b.size(), real(0))};
}

/// @brief Build a solve cache seeded with an initial guess u0 (warm start).
template <class Op, class Alg>
LinearCache<Op, Alg> init(const LinearProblem<Op> &prob, const Alg &alg, Vector u0) {
    return {prob.A, prob.b, alg, std::move(u0)};
}

/// @brief Re-solve from a cache, warm-starting from its current iterate; cache.u
/// is updated in place (the C++ spelling of CommonSolve `solve!`).
template <class Op, class Alg>
LinearSolution solve(LinearCache<Op, Alg> &cache) {
    const SolverResult r = detail::run(cache.A, cache.b, cache.u, cache.alg);
    return {cache.u, r.iterations, r.residual, r.converged};
}

/// @brief One-shot linear solve: solve(problem, algorithm) == solve(init(...)).
template <class Op, class Alg>
LinearSolution solve(const LinearProblem<Op> &prob, const Alg &alg) {
    Vector u(prob.b.size(), real(0));
    const SolverResult r = detail::run(prob.A, prob.b, u, alg);
    return {std::move(u), r.iterations, r.residual, r.converged};
}

} // namespace num
