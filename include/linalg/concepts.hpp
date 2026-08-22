/// @file linalg/concepts.hpp
/// @brief Linear algebra solver, preconditioner, and factorization concepts.
#pragma once

#include "core/concepts.hpp"
#include "linalg/solvers/solver_result.hpp"
#include "operator/concepts.hpp"
#include <concepts>

namespace num {

/// @brief Preconditioner interface supporting approximate inversion z = M^-1 r.
template <class M, class X = Vector, class Y = Vector>
concept Preconditioner = requires(const M &M_op, const X &r, Y &z) {
    { M_op.rows() } -> std::convertible_to<idx>;
    { M_op.cols() } -> std::convertible_to<idx>;
    M_op.apply(r, z);
};

/// @brief Factorization object supporting direct triangular solves.
template <class F, class Vec = Vector>
concept TriangularFactor = requires(const F &factor, const Vec &b, Vec &x) {
    { factor.solve(b, x) };
};

/// @brief Linear solver contract returning convergence diagnostics.
template <class S, class Op, class Vec = Vector>
concept IsLinearSolver = requires(const S &solver, const Op &A, const Vec &b, Vec &x) {
    { solver.solve(A, b, x) } -> std::same_as<SolverResult>;
};

} // namespace num
