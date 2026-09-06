/// @file linear/solvers/hessenberg_resolvent.hpp
/// @brief O(n^2)-per-shift complex resolvent solver based on Hessenberg decomposition.
#pragma once

#include "core/debug.hpp"
#include "container/matrix.hpp"
#include "core/types.hpp"
#include "container/vector.hpp"
#include "linear/factorization/hessenberg.hpp"
#include <complex>
#include <vector>

namespace num {

/// @brief High-performance resolvent solver (sI - A)^-1 b reusing a precomputed Hessenberg decomposition.
///
/// Preprocessing cost: O(n^3) once (Hessenberg reduction A = Q H Q^T).
/// Per-shift solve cost: O(n^2) (Hessenberg-structured Gaussian elimination + Q back-projection).
class hessenberg_resolvent_solver {
  public:
    /// Construct and decompose A into upper Hessenberg form.
    explicit hessenberg_resolvent_solver(const mat &A);
    explicit hessenberg_resolvent_solver(hessenberg_decomposition decomp);

    [[nodiscard]] idx size() const noexcept { return decomp_.size(); }
    [[nodiscard]] const hessenberg_decomposition &decomposition() const noexcept { return decomp_; }

    /// Solve (sI - A) x = b for a single shift and real RHS in O(n^2).
    [[nodiscard]] std::vector<cplx> solve(cplx shift, const vec &b) const;

    /// Solve (sI - A) x = b for a single shift and complex RHS in O(n^2).
    [[nodiscard]] std::vector<cplx> solve(cplx shift, const std::vector<cplx> &b) const;

    /// Solve for multiple shifts and a single RHS in parallel O(n^3 + k * n^2).
    [[nodiscard]] std::vector<std::vector<cplx>>
    solve_batch(const std::vector<cplx> &shifts, const vec &b) const;

    /// Solve for multiple shifts and multiple RHS vectors in parallel.
    [[nodiscard]] std::vector<std::vector<std::vector<cplx>>>
    solve_batch(const std::vector<cplx> &shifts, const std::vector<vec> &rhs_list) const;

  private:
    hessenberg_decomposition decomp_;
};

} // namespace num
