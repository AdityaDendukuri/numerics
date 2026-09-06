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
struct sparse_resolvent_options {
    bool symmetric_pattern = false;
};

/// Reusable sparse solver for (s I - A).  With SuiteSparse enabled, the
/// sparsity analysis is retained while numeric values are rebuilt per shift.
/// Without SuiteSparse, factorize/solve report that no sparse complex backend
/// is available rather than silently densifying a large matrix.
class sparse_resolvent_solver {
  public:
    /// Analyze A's sparsity pattern once for repeated numerical factorizations.
    explicit sparse_resolvent_solver(const spmat &A, sparse_resolvent_options options = {});
    ~sparse_resolvent_solver();
    sparse_resolvent_solver(sparse_resolvent_solver &&) noexcept;
    sparse_resolvent_solver &operator=(sparse_resolvent_solver &&) noexcept;
    sparse_resolvent_solver(const sparse_resolvent_solver &) = delete;
    sparse_resolvent_solver &operator=(const sparse_resolvent_solver &) = delete;

    /// Return the order of A.
    [[nodiscard]] idx size() const noexcept;
    /// Numerically factor sI-A while retaining symbolic analysis.
    void factorize(cplx shift);
    /// Solve one or more right-hand sides against the current shift.
    [[nodiscard]] array<cplx> solve(const array<cplx> &rhs) const;
    void solve(const array<cplx> &rhs, array<cplx> &out) const;
    [[nodiscard]] array<array<cplx>>
    solve(const array<array<cplx>> &rhs) const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace num
