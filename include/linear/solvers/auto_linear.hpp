/// @file linear/solvers/auto_linear.hpp
/// @brief Automatic dense/sparse factorization for reusable real solves.
#pragma once

#include "container/matrix.hpp"
#include "container/vector.hpp"
#include "linear/sparse/sparse.hpp"
#include <memory>

namespace num {

/// Select dense LU at or below `dense_limit`, otherwise prefer SuiteSparse KLU.
struct auto_linear_options {
    idx dense_limit = 32;
};

/// Reusable real factorization that selects a dense or sparse backend by size.
class auto_linear_solver {
  public:
    /// Factor a square CSR matrix using the configured backend threshold.
    explicit auto_linear_solver(const spmat &matrix, auto_linear_options options = {});
    ~auto_linear_solver();
    auto_linear_solver(auto_linear_solver &&) noexcept;
    auto_linear_solver &operator=(auto_linear_solver &&) noexcept;
    auto_linear_solver(const auto_linear_solver &) = delete;
    auto_linear_solver &operator=(const auto_linear_solver &) = delete;

    /// Return the order of the factored matrix, or zero after a move.
    [[nodiscard]] idx size() const noexcept;
    /// Solve AX=B without modifying the stored factorization.
    void solve(const vec &rhs, vec &solution) const;
    void solve(const mat &rhs, mat &solution) const;
    /// Solve A^T x=b without modifying the stored factorization.
    void solve_transpose(const vec &rhs, vec &solution) const;
    /// Solve A^T X=B for several dense right-hand sides.
    void solve_transpose(const mat &rhs, mat &solution) const;
    /// Replace one or more right-hand sides with their solutions.
    void solve_in_place(vec &right_hand_side) const;
    void solve_in_place(mat &right_hand_sides) const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// Convenience solve overload. Allocates; prefer the out-param form in hot loops.
[[nodiscard]] inline vec solve(const auto_linear_solver &factor, const vec &rhs) {
    vec solution(rhs.size(), 0.0);
    factor.solve(rhs, solution);
    return solution;
}

/// Convenience solve overload for several dense right-hand sides.
/// Allocates; prefer the out-param form in hot loops.
[[nodiscard]] inline mat solve(const auto_linear_solver &factor, const mat &rhs) {
    mat solution;
    factor.solve(rhs, solution);
    return solution;
}

/// Convenience transpose solve overload. Allocates; prefer the out-param form in hot loops.
[[nodiscard]] inline vec solve_transpose(const auto_linear_solver &factor, const vec &rhs) {
    vec solution(rhs.size(), 0.0);
    factor.solve_transpose(rhs, solution);
    return solution;
}

/// Convenience transpose solve overload for several dense right-hand sides.
/// Allocates; prefer the out-param form in hot loops.
[[nodiscard]] inline mat solve_transpose(const auto_linear_solver &factor, const mat &rhs) {
    mat solution;
    factor.solve_transpose(rhs, solution);
    return solution;
}

} // namespace num
