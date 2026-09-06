/// @file dense_resolvent.hpp
/// @brief Dense solver for repeatedly shifted complex linear systems.
#pragma once

#include "container/matrix.hpp"
#include "core/types.hpp"
#include "linear/sparse/sparse.hpp"
#include <memory>
#include <vector>

namespace num {

/// Reusable dense solver for (s I - A)x = b.
class dense_resolvent_solver {
  public:
    /// Store A densely for repeated shifts.
    explicit dense_resolvent_solver(const mat &matrix);
    explicit dense_resolvent_solver(const spmat &matrix);
    ~dense_resolvent_solver();
    dense_resolvent_solver(dense_resolvent_solver &&) noexcept;
    dense_resolvent_solver &operator=(dense_resolvent_solver &&) noexcept;
    dense_resolvent_solver(const dense_resolvent_solver &) = delete;
    dense_resolvent_solver &operator=(const dense_resolvent_solver &) = delete;

    /// Return the order of A.
    [[nodiscard]] idx size() const noexcept;
    /// Factor sI-A for subsequent solves.
    void factorize(cplx shift);
    /// Solve the currently factored shifted system.
    [[nodiscard]] array<cplx> solve(const array<cplx> &rhs) const;
    void solve(const array<cplx> &rhs, array<cplx> &result) const;
    /// Solve several right-hand sides against the current shift.
    [[nodiscard]] array<array<cplx>>
    solve(const array<array<cplx>> &right_hand_sides) const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace num
