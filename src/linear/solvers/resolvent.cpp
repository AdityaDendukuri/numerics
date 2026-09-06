/// @file linear/solvers/resolvent.cpp
/// @brief Resolvent convenience interfaces powered by Hessenberg decomposition.
#include "linear/solvers/resolvent.hpp"
#include "core/debug.hpp"

namespace num {
namespace {

[[nodiscard]] std::vector<cplx> complex_copy(const vec &source) {
    std::vector<cplx> result(source.size());
    for (idx index = 0; index < source.size(); ++index) {
        result[index] = source[index];
    }
    return result;
}

[[nodiscard]] std::vector<std::vector<cplx>> complex_copy(const std::vector<vec> &sources) {
    std::vector<std::vector<cplx>> result;
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

std::vector<cplx> resolvent_factor::solve(const std::vector<cplx> &rhs) const {
    debug::check_dim(solver_.size(), static_cast<idx>(rhs.size()), "resolvent_factor RHS");
    return solver_.solve(rhs);
}

std::vector<std::vector<cplx>>
resolvent_factor::solve(const std::vector<std::vector<cplx>> &right_hand_sides) const {
    return solver_.solve(right_hand_sides);
}

std::vector<cplx> resolvent_solve(cplx shift, const mat &matrix, const vec &right_hand_side) {
    debug::check_dim(matrix.rows(), right_hand_side.size(), "resolvent_solve RHS");
    hessenberg_resolvent_solver solver(matrix);
    return solver.solve(shift, right_hand_side);
}

std::vector<std::vector<cplx>>
resolvent_solve_rhs_batch(cplx shift, const mat &matrix,
                          const std::vector<vec> &right_hand_sides) {
    resolvent_factor factor(shift, matrix);
    return factor.solve(complex_copy(right_hand_sides));
}

std::vector<std::vector<cplx>> resolvent_solve_batch(const std::vector<cplx> &shifts,
                                                     const mat &matrix,
                                                     const vec &right_hand_side) {
    debug::check_dim(matrix.rows(), matrix.cols(), "resolvent_solve_batch matrix must be square");
    debug::check_dim(matrix.rows(), right_hand_side.size(), "resolvent_solve_batch RHS");
    debug::check_non_empty(matrix.rows(), "resolvent_solve_batch matrix");

    // O(n^3) Hessenberg reduction once + O(k * n^2) parallel Hessenberg solves
    hessenberg_resolvent_solver solver(matrix);
    return solver.solve_batch(shifts, right_hand_side);
}

std::vector<std::vector<std::vector<cplx>>>
resolvent_solve_batch(const std::vector<cplx> &shifts, const mat &matrix,
                      const std::vector<vec> &right_hand_sides) {
    debug::check_dim(matrix.rows(), matrix.cols(), "resolvent_solve_batch matrix must be square");
    debug::check_non_empty(matrix.rows(), "resolvent_solve_batch matrix");

    // O(n^3) Hessenberg reduction once + O(k * m * n^2) parallel Hessenberg solves
    hessenberg_resolvent_solver solver(matrix);
    return solver.solve_batch(shifts, right_hand_sides);
}

} // namespace num
