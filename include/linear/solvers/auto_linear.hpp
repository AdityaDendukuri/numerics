/// @file linear/solvers/auto_linear.hpp
/// @brief Reusable factorization, dense or sparse by size.
#pragma once

#include "container/matrix.hpp"
#include "container/vector.hpp"
#include "linear/solve.hpp"
#include "linear/sparse/sparse.hpp"
#include <memory>

namespace num {

/// @brief Backend selection: dense LU up to `dense_limit`, else sparse (KLU).
struct auto_linear_options {
    idx dense_limit = 32;
};

/// @brief Automatic factorization: dense LU or sparse LU by size.
class auto_linear_solver {
  public:
    /// @brief Factor a square sparse matrix.
    explicit auto_linear_solver(const spmat &matrix, auto_linear_options options = {});
    ~auto_linear_solver();
    auto_linear_solver(auto_linear_solver &&) noexcept;
    auto_linear_solver &operator=(auto_linear_solver &&) noexcept;
    auto_linear_solver(const auto_linear_solver &) = delete;
    auto_linear_solver &operator=(const auto_linear_solver &) = delete;

    /// @brief Order \f$n\f$, or 0 after a move.
    [[nodiscard]] idx size() const noexcept;

  private:
    friend void solve(const auto_linear_solver &, const vec<real> &, vec<real> &);
    friend void solve(const auto_linear_solver &, const mat<real> &, mat<real> &);
    friend void solve_transpose(const auto_linear_solver &, const vec<real> &, vec<real> &);
    friend void solve_transpose(const auto_linear_solver &, const mat<real> &, mat<real> &);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// @brief Solve: \f$x \leftarrow A^{-1} b\f$. `x` may alias `b`.
void solve(const auto_linear_solver &factor, const vec<real> &b, vec<real> &x);
/// @brief Solve: \f$X \leftarrow A^{-1} B\f$. `X` may alias `B`.
void solve(const auto_linear_solver &factor, const mat<real> &B, mat<real> &X);
/// @brief Transposed solve: \f$x \leftarrow A^{-T} b\f$. `x` may alias `b`.
void solve_transpose(const auto_linear_solver &factor, const vec<real> &b, vec<real> &x);
void solve_transpose(const auto_linear_solver &factor, const mat<real> &B, mat<real> &X);

} // namespace num
