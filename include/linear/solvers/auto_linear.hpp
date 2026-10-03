/// @file linear/solvers/auto_linear.hpp
/// @brief Automatic dense/sparse factorization for reusable real solves.
#pragma once

#include "container/matrix.hpp"
#include "container/vector.hpp"
#include "linear/solve.hpp"
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

  private:
    friend void solve(const auto_linear_solver &, const vec<real> &, vec<real> &);
    friend void solve(const auto_linear_solver &, const mat<real> &, mat<real> &);
    friend void solve_transpose(const auto_linear_solver &, const vec<real> &, vec<real> &);
    friend void solve_transpose(const auto_linear_solver &, const mat<real> &, mat<real> &);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// @brief Solve \f$Ax = b\f$. `x` may be `b`.
void solve(const auto_linear_solver &factor, const vec<real> &b, vec<real> &x);
/// @brief Solve \f$AX = B\f$. `X` may be `B`.
void solve(const auto_linear_solver &factor, const mat<real> &B, mat<real> &X);
/// @brief Solve \f$A^Tx = b\f$. `x` may be `b`.
void solve_transpose(const auto_linear_solver &factor, const vec<real> &b, vec<real> &x);
void solve_transpose(const auto_linear_solver &factor, const mat<real> &B, mat<real> &X);

} // namespace num
