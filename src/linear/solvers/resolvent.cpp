/// @file linear/solvers/resolvent.cpp
/// @brief Resolvent convenience interfaces powered by Hessenberg decomposition.
#include "linear/solvers/resolvent.hpp"
#include "core/debug.hpp"

namespace num {
namespace {

[[nodiscard]] array<cplx> complex_copy(const vec &source) {
    array<cplx> result(source.size());
    for (idx index = 0; index < source.size(); ++index) {
        result[index] = source[index];
    }
    return result;
}

[[nodiscard]] array<array<cplx>> complex_copy(const array<vec> &sources) {
    array<array<cplx>> result;
    result.reserve(sources.size());
    for (const vec &source : sources) {
        result.push_back(complex_copy(source));
    }
    return result;
}

} // namespace

resolvent_factor::resolvent_factor(cplx shift, const mat &matrix) : solver_(matrix) {
    debug::check_dim(matrix.rows(), matrix.cols(), "resolvent_factor matrix must be square");
    debug::check_non_empty(matrix.rows(), "resolvent_factor matrix");
    solver_.factorize(shift);
}

array<cplx> resolvent_factor::solve(const array<cplx> &rhs) const {
    debug::check_dim(solver_.size(), static_cast<idx>(rhs.size()), "resolvent_factor RHS");
    return solver_.solve(rhs);
}

array<array<cplx>>
resolvent_factor::solve(const array<array<cplx>> &right_hand_sides) const {
    return solver_.solve(right_hand_sides);
}

array<cplx> resolvent_solve(cplx shift, const mat &matrix, const vec &right_hand_side) {
    debug::check_dim(matrix.rows(), right_hand_side.size(), "resolvent_solve RHS");
    hessenberg_resolvent_solver solver(matrix);
    return solver.solve(shift, right_hand_side);
}

array<array<cplx>>
resolvent_solve_rhs_batch(cplx shift, const mat &matrix,
                          const array<vec> &right_hand_sides) {
    resolvent_factor factor(shift, matrix);
    return factor.solve(complex_copy(right_hand_sides));
}

array<array<cplx>> resolvent_solve_batch(const array<cplx> &shifts,
                                                     const mat &matrix,
                                                     const vec &right_hand_side) {
    debug::check_dim(matrix.rows(), matrix.cols(), "resolvent_solve_batch matrix must be square");
    debug::check_dim(matrix.rows(), right_hand_side.size(), "resolvent_solve_batch RHS");
    debug::check_non_empty(matrix.rows(), "resolvent_solve_batch matrix");

    // O(n^3) Hessenberg reduction once + O(k * n^2) parallel Hessenberg solves
    hessenberg_resolvent_solver solver(matrix);
    return solver.solve_batch(shifts, right_hand_side);
}

array<array<array<cplx>>>
resolvent_solve_batch(const array<cplx> &shifts, const mat &matrix,
                      const array<vec> &right_hand_sides) {
    debug::check_dim(matrix.rows(), matrix.cols(), "resolvent_solve_batch matrix must be square");
    debug::check_non_empty(matrix.rows(), "resolvent_solve_batch matrix");

    // O(n^3) Hessenberg reduction once + O(k * m * n^2) parallel Hessenberg solves
    hessenberg_resolvent_solver solver(matrix);
    return solver.solve_batch(shifts, right_hand_sides);
}

} // namespace num
