/// @file linear/solvers/resolvent.hpp
/// @brief Convenience functions for dense shifted systems (sI-A)x=b.
#pragma once

#include "core/types.hpp"
#include "container/matrix.hpp"
#include "container/vector.hpp"
#include "linear/solvers/dense_resolvent.hpp"
#include "linear/solvers/hessenberg_resolvent.hpp"
#include <vector>

namespace num {

/// Compatibility adapter that factors one dense shift during construction.
class resolvent_factor {
  public:
    /// Factor sI-A for repeated right-hand sides.
    resolvent_factor(cplx s, const mat &A);

    /// Solve one complex right-hand side.
    [[nodiscard]] array<cplx> solve(const array<cplx> &rhs) const;

    /// Solve several complex right-hand sides with the same factorization.
    [[nodiscard]] array<array<cplx>>
    solve(const array<array<cplx>> &right_hand_sides) const;

  private:
    dense_resolvent_solver solver_;
};

/// Solve one dense shifted system with a real right-hand side.
[[nodiscard]] array<cplx> resolvent_solve(cplx shift, const mat &matrix,
                                                const vec &right_hand_side);

/// Solve one dense shift for several real right-hand sides.
[[nodiscard]] array<array<cplx>>
resolvent_solve_rhs_batch(cplx shift, const mat &matrix,
                          const array<vec> &right_hand_sides);

/// Solve several dense shifts for one real right-hand side.
[[nodiscard]] array<array<cplx>> resolvent_solve_batch(const array<cplx> &shifts,
                                                                   const mat &matrix,
                                                                   const vec &right_hand_side);

/// Solve several dense shifts for several real right-hand sides.
[[nodiscard]] array<array<array<cplx>>>
resolvent_solve_batch(const array<cplx> &shifts, const mat &matrix,
                      const array<vec> &right_hand_sides);

} // namespace num
