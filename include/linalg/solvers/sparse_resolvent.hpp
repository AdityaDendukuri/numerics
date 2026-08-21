/// @file sparse_resolvent.hpp
/// @brief Shifted sparse resolvent plans with optional SuiteSparse backend.
#pragma once

#include "core/types.hpp"
#include "linalg/sparse/sparse.hpp"
#include <memory>
#include <vector>

namespace num {

[[nodiscard]] bool sparse_resolvent_available() noexcept;

struct SparseResolventOptions {
  bool symmetric_pattern = false;
};

/// Reusable sparse solver for (s I - A).  With SuiteSparse enabled, the
/// sparsity analysis is retained while numeric values are rebuilt per shift.
/// Without SuiteSparse, factorize/solve report that no sparse complex backend
/// is available rather than silently densifying a large matrix.
class SparseResolventSolver {
public:
  explicit SparseResolventSolver(const SparseMatrix& A,
                                 SparseResolventOptions options = {});
  ~SparseResolventSolver();
  SparseResolventSolver(SparseResolventSolver&&) noexcept;
  SparseResolventSolver& operator=(SparseResolventSolver&&) noexcept;
  SparseResolventSolver(const SparseResolventSolver&) = delete;
  SparseResolventSolver& operator=(const SparseResolventSolver&) = delete;

  [[nodiscard]] idx size() const noexcept;
  void factorize(cplx shift);
  [[nodiscard]] std::vector<cplx> solve(const std::vector<cplx>& rhs) const;
  void solve(const std::vector<cplx>& rhs, std::vector<cplx>& out) const;
  [[nodiscard]] std::vector<std::vector<cplx>> solve(
    const std::vector<std::vector<cplx>>& rhs) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace num
