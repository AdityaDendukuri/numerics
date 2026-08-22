/// @file src/linalg/solvers/resolvent.cpp
/// @brief Resolvent convenience interfaces powered by Hessenberg decomposition.
#include "linalg/solvers/resolvent.hpp"
#include "core/debug.hpp"

namespace num {
namespace {

[[nodiscard]] std::vector<cplx> complex_copy(const Vector &source) {
    std::vector<cplx> result(source.size());
    for (idx index = 0; index < source.size(); ++index) {
        result[index] = source[index];
    }
    return result;
}

[[nodiscard]] std::vector<std::vector<cplx>> complex_copy(const std::vector<Vector> &sources) {
    std::vector<std::vector<cplx>> result;
    result.reserve(sources.size());
    for (const Vector &source : sources) {
        result.push_back(complex_copy(source));
    }
    return result;
}

} // namespace

ResolventFactor::ResolventFactor(cplx shift, const Matrix &matrix) : solver_(matrix) {
    debug::check_dim(matrix.rows(), matrix.cols(), "ResolventFactor matrix must be square");
    debug::check_non_empty(matrix.rows(), "ResolventFactor matrix");
    solver_.factorize(shift);
}

std::vector<cplx> ResolventFactor::solve(const std::vector<cplx> &rhs) const {
    debug::check_dim(solver_.size(), static_cast<idx>(rhs.size()), "ResolventFactor RHS");
    return solver_.solve(rhs);
}

std::vector<std::vector<cplx>>
ResolventFactor::solve(const std::vector<std::vector<cplx>> &right_hand_sides) const {
    return solver_.solve(right_hand_sides);
}

std::vector<cplx> resolvent_solve(cplx shift, const Matrix &matrix, const Vector &right_hand_side) {
    debug::check_dim(matrix.rows(), right_hand_side.size(), "resolvent_solve RHS");
    HessenbergResolventSolver solver(matrix);
    return solver.solve(shift, right_hand_side);
}

std::vector<std::vector<cplx>>
resolvent_solve_rhs_batch(cplx shift, const Matrix &matrix,
                          const std::vector<Vector> &right_hand_sides) {
    ResolventFactor factor(shift, matrix);
    return factor.solve(complex_copy(right_hand_sides));
}

std::vector<std::vector<cplx>> resolvent_solve_batch(const std::vector<cplx> &shifts,
                                                     const Matrix &matrix,
                                                     const Vector &right_hand_side) {
    debug::check_dim(matrix.rows(), matrix.cols(), "resolvent_solve_batch matrix must be square");
    debug::check_dim(matrix.rows(), right_hand_side.size(), "resolvent_solve_batch RHS");
    debug::check_non_empty(matrix.rows(), "resolvent_solve_batch matrix");

    // O(n^3) Hessenberg reduction once + O(k * n^2) parallel Hessenberg solves
    HessenbergResolventSolver solver(matrix);
    return solver.solve_batch(shifts, right_hand_side);
}

std::vector<std::vector<std::vector<cplx>>>
resolvent_solve_batch(const std::vector<cplx> &shifts, const Matrix &matrix,
                      const std::vector<Vector> &right_hand_sides) {
    debug::check_dim(matrix.rows(), matrix.cols(), "resolvent_solve_batch matrix must be square");
    debug::check_non_empty(matrix.rows(), "resolvent_solve_batch matrix");

    // O(n^3) Hessenberg reduction once + O(k * m * n^2) parallel Hessenberg solves
    HessenbergResolventSolver solver(matrix);
    return solver.solve_batch(shifts, right_hand_sides);
}

} // namespace num
