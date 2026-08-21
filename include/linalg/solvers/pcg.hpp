/// @file solvers/pcg.hpp
/// @brief Preconditioned conjugate gradient.
#pragma once

#include "core/policy.hpp"
#include "core/vector.hpp"
#include "linalg/solvers/preconditioner.hpp"
#include "linalg/solvers/solver_result.hpp"
#include "operator/concepts.hpp"
#include <cmath>
#include <stdexcept>

namespace num {

template<class Op, class M>
  requires SPDLinearOperator<Op, Vector, Vector> && Preconditioner<M>
/// Solve an SPD operator system with a supplied preconditioner.
SolverResult pcg(const Op& A,
                 const M& M_op,
                 const Vector& b,
                 Vector& x,
                 real tol = 1e-10,
                 idx max_iter = 1000,
                 Backend backend = default_backend) {
  const idx n = b.size();
  if (A.rows() != n || A.cols() != n || M_op.rows() != n || M_op.cols() != n
      || x.size() != n) {
    throw std::invalid_argument("pcg: dimension mismatch");
  }

  Vector r(n), z(n), p(n), Ap(n);
  A.apply(x, r);
  for (idx i = 0; i < n; ++i) {
    r[i] = b[i] - r[i];
  }
  M_op.apply(r, z);
  p = z;

  real rzold = dot(r, z, backend);
  SolverResult result{0, norm(r, backend), false};

  for (idx iter = 0; iter < max_iter; ++iter) {
    result.iterations = iter + 1;
    A.apply(p, Ap);

    const real pAp = dot(p, Ap, backend);
    if (std::abs(pAp) < real(1e-15)) {
      break;
    }

    const real alpha = rzold / pAp;
    axpy(alpha, p, x, backend);
    axpy(-alpha, Ap, r, backend);

    result.residual = norm(r, backend);
    if (result.residual < tol) {
      result.converged = true;
      break;
    }

    M_op.apply(r, z);
    const real rznew = dot(r, z, backend);
    const real beta = rznew / rzold;
    scale(p, beta, backend);
    axpy(real(1), z, p, backend);
    rzold = rznew;
  }

  return result;
}

} // namespace num
