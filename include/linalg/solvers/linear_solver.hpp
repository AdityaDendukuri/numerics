/// @file linalg/solvers/linear_solver.hpp
/// @brief Universal linear solver callable type.
///
/// LinearSolver is a first-class callable: given a right-hand side vector,
/// it solves A*x = rhs in-place.  Any algorithm (CG, LU, Cholesky, ...) that
/// satisfies this contract can be stored as a LinearSolver and passed to ODE
/// integrators, PDE time-steppers, or nonlinear solvers without change.
///
/// Factory functions in linalg/solvers/ return LinearSolver:
///   num::make_cg_solver(A)
///   num::make_lu_solver(A)    (planned)
///   num::make_chol_solver(A)  (planned)
#pragma once

#include "core/vector.hpp"
#include "linalg/solvers/solver_result.hpp"
#include <functional>

namespace num {

/// Callable that solves A*x = rhs in-place.  rhs is passed by value so
/// the solver can use it as a work vector without copying.
using LinearSolver = std::function<SolverResult(const Vector& rhs, Vector& x)>;

} // namespace num
