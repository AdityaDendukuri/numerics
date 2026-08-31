/// @file sparse_resolvent.hpp
/// @brief Shifted sparse resolvent plans with optional SuiteSparse backend.
#pragma once

#include "core/types.hpp"
#include "linear/sparse/sparse.hpp"
#include <memory>
#include <vector>

namespace num {

/// True when a sparse complex factorization backend is available.
[[nodiscard]] bool sparse_resolvent_available() noexcept;

/// Symbolic-analysis hints for sparse shifted systems.
struct SparseResolventOptions {
    bool symmetric_pattern = false;
};

/// Reusable sparse solver for (s I - A).  With SuiteSparse enabled, the
/// sparsity analysis is retained while numeric values are rebuilt per shift.
/// Without SuiteSparse, factorize/solve report that no sparse complex backend
/// is available rather than silently densifying a large matrix.
class SparseResolventSolver {
  public:
    /// Analyze A's sparsity pattern once for repeated numerical factorizations.
    explicit SparseResolventSolver(const SparseMatrix &A, SparseResolventOptions options = {});
    ~SparseResolventSolver();
    SparseResolventSolver(SparseResolventSolver &&) noexcept;
    SparseResolventSolver &operator=(SparseResolventSolver &&) noexcept;
    SparseResolventSolver(const SparseResolventSolver &) = delete;
    SparseResolventSolver &operator=(const SparseResolventSolver &) = delete;

    /// Return the order of A.
    [[nodiscard]] idx size() const noexcept;
    /// Numerically factor sI-A while retaining symbolic analysis.
    void factorize(cplx shift);
    /// Solve one or more right-hand sides against the current shift.
    [[nodiscard]] std::vector<cplx> solve(const std::vector<cplx> &rhs) const;
    void solve(const std::vector<cplx> &rhs, std::vector<cplx> &out) const;
    [[nodiscard]] std::vector<std::vector<cplx>>
    solve(const std::vector<std::vector<cplx>> &rhs) const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace num
