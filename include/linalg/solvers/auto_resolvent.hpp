/// @file linalg/solvers/auto_resolvent.hpp
/// @brief Automatic dense/sparse shifted-resolvent selection.
#pragma once

#include "core/types.hpp"
#include "linalg/solvers/sparse_resolvent.hpp"
#include "linalg/sparse/sparse.hpp"
#include <memory>
#include <vector>

namespace num {

struct AutoResolventOptions {
  idx dense_limit = 128;
  bool symmetric_pattern = false;
};

class AutoResolventSolver {
public:
  explicit AutoResolventSolver(const SparseMatrix& matrix,
                               AutoResolventOptions options = {});
  ~AutoResolventSolver();
  AutoResolventSolver(AutoResolventSolver&&) noexcept;
  AutoResolventSolver& operator=(AutoResolventSolver&&) noexcept;
  AutoResolventSolver(const AutoResolventSolver&) = delete;
  AutoResolventSolver& operator=(const AutoResolventSolver&) = delete;

  [[nodiscard]] idx size() const noexcept;
  void factorize(cplx shift);
  void solve(const std::vector<cplx>& rhs, std::vector<cplx>& result) const;
  [[nodiscard]] std::vector<std::vector<cplx>> solve(
    const std::vector<std::vector<cplx>>& right_hand_sides) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace num
