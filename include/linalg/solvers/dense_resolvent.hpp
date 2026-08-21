/// @file dense_resolvent.hpp
/// @brief Dense solver for repeatedly shifted complex linear systems.
#pragma once

#include "core/matrix.hpp"
#include "core/types.hpp"
#include "linalg/sparse/sparse.hpp"
#include <memory>
#include <vector>

namespace num {

/// Reusable dense solver for (s I - A)x = b.
class DenseResolventSolver {
public:
  explicit DenseResolventSolver(const Matrix& matrix);
  explicit DenseResolventSolver(const SparseMatrix& matrix);
  ~DenseResolventSolver();
  DenseResolventSolver(DenseResolventSolver&&) noexcept;
  DenseResolventSolver& operator=(DenseResolventSolver&&) noexcept;
  DenseResolventSolver(const DenseResolventSolver&) = delete;
  DenseResolventSolver& operator=(const DenseResolventSolver&) = delete;

  [[nodiscard]] idx size() const noexcept;
  void factorize(cplx shift);
  [[nodiscard]] std::vector<cplx> solve(const std::vector<cplx>& rhs) const;
  void solve(const std::vector<cplx>& rhs, std::vector<cplx>& result) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace num
