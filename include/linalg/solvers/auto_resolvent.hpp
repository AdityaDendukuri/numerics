/// @file linalg/solvers/auto_resolvent.hpp
/// @brief Automatic dense/sparse shifted-resolvent selection.
#pragma once

#include "core/types.hpp"
#include "linalg/solvers/sparse_resolvent.hpp"
#include "linalg/sparse/sparse.hpp"
#include <memory>
#include <vector>

namespace num {

/// Dense/sparse cutoff and optional sparse symbolic symmetry hint.
struct AutoResolventOptions {
    idx dense_limit = 512;
    bool symmetric_pattern = false;
};

/// Reusable solver for shifted systems (zI-A)x=b with automatic backend choice.
class AutoResolventSolver {
  public:
    /// Store A and select the dense or sparse shifted-system implementation.
    explicit AutoResolventSolver(const SparseMatrix &matrix, AutoResolventOptions options = {});
    ~AutoResolventSolver();
    AutoResolventSolver(AutoResolventSolver &&) noexcept;
    AutoResolventSolver &operator=(AutoResolventSolver &&) noexcept;
    AutoResolventSolver(const AutoResolventSolver &) = delete;
    AutoResolventSolver &operator=(const AutoResolventSolver &) = delete;

    /// Return the order of A.
    [[nodiscard]] idx size() const noexcept;
    /// Factor the shifted matrix zI-A for subsequent solves.
    void factorize(cplx shift);
    /// Solve the currently factored shifted system.
    void solve(const std::vector<cplx> &rhs, std::vector<cplx> &result) const;
    /// Solve several right-hand sides against the current shift.
    [[nodiscard]] std::vector<std::vector<cplx>>
    solve(const std::vector<std::vector<cplx>> &right_hand_sides) const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace num
