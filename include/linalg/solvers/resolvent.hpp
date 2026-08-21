/// @file linalg/solvers/resolvent.hpp
/// @brief Convenience functions for dense shifted systems (sI-A)x=b.
#pragma once

#include "core/matrix.hpp"
#include "core/vector.hpp"
#include "linalg/solvers/dense_resolvent.hpp"
#include <vector>

namespace num {

/// Compatibility adapter that factors one dense shift during construction.
class ResolventFactor {
public:
  /// Factor sI-A for repeated right-hand sides.
  ResolventFactor(cplx s, const Matrix& A);

  /// Solve one complex right-hand side.
  [[nodiscard]] std::vector<cplx> solve(const std::vector<cplx>& rhs) const;

  /// Solve several complex right-hand sides with the same factorization.
  [[nodiscard]] std::vector<std::vector<cplx>> solve(
    const std::vector<std::vector<cplx>>& right_hand_sides) const;

private:
  DenseResolventSolver solver_;
};

/// Solve one dense shifted system with a real right-hand side.
[[nodiscard]] std::vector<cplx> resolvent_solve(cplx shift,
                                                const Matrix& matrix,
                                                const Vector& right_hand_side);

/// Solve one dense shift for several real right-hand sides.
[[nodiscard]] std::vector<std::vector<cplx>> resolvent_solve_rhs_batch(
  cplx shift,
  const Matrix& matrix,
  const std::vector<Vector>& right_hand_sides);

/// Solve several dense shifts for one real right-hand side.
[[nodiscard]] std::vector<std::vector<cplx>> resolvent_solve_batch(
  const std::vector<cplx>& shifts,
  const Matrix& matrix,
  const Vector& right_hand_side);

/// Solve several dense shifts for several real right-hand sides.
[[nodiscard]] std::vector<std::vector<std::vector<cplx>>> resolvent_solve_batch(
  const std::vector<cplx>& shifts,
  const Matrix& matrix,
  const std::vector<Vector>& right_hand_sides);

} // namespace num
