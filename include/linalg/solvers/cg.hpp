/// @file cg.hpp
/// @brief Conjugate gradient solvers.
///
/// Solves \f$Ax=b\f$ for symmetric positive definite \f$A\f$ using
/// \f$\mathcal{K}_k(A,r_0)=\mathrm{span}\{r_0,Ar_0,\ldots,A^{k-1}r_0\}\f$.
#pragma once
#include "core/matrix.hpp"
#include "core/policy.hpp"
#include "core/vector.hpp"
#include "linalg/matrix_properties.hpp"
#include "linalg/solvers/solver_result.hpp"
#include "core/concepts.hpp"
#include <cmath>
#include <stdexcept>

namespace num {

SolverResult cg(const Matrix& A,
                const Vector& b,
                Vector& x,
                real tol = 1e-10,
                idx max_iter = 1000,
                Backend backend = default_backend);

inline SolverResult cg(const linalg::SPDMatrix<Matrix>& A,
                       const Vector& b,
                       Vector& x,
                       real tol = 1e-10,
                       idx max_iter = 1000,
                       Backend backend = default_backend) {
  return cg(A.base(), b, x, tol, max_iter, backend);
}

namespace detail {

template<class Op>
  requires LinearOperator<Op, Vector, Vector>
SolverResult cg_operator_impl(const Op& A,
                              const Vector& b,
                              Vector& x,
                              real tol,
                              idx max_iter,
                              Backend backend) {
  const idx n = b.size();
  if (A.rows() != n || A.cols() != n || x.size() != n) {
    throw std::invalid_argument("Dimension mismatch in operator CG solver");
  }

  Vector r(n), p(n), Ap(n);
  A.apply(x, r);
  for (idx i = 0; i < n; ++i) {
    r[i] = b[i] - r[i];
    p[i] = r[i];
  }

  real rsold = dot(r, r, backend);
  SolverResult result{0, std::sqrt(rsold), false};

  for (idx iter = 0; iter < max_iter; ++iter) {
    result.iterations = iter + 1;
    A.apply(p, Ap);

    const real pAp = dot(p, Ap, backend);
    if (std::abs(pAp) < real(1e-15)) {
      break;
    }
    const real alpha = rsold / pAp;

    axpy(alpha, p, x, backend);
    axpy(-alpha, Ap, r, backend);

    const real rsnew = dot(r, r, backend);
    result.residual = std::sqrt(rsnew);
    if (result.residual < tol) {
      result.converged = true;
      break;
    }

    const real beta = rsnew / rsold;
    scale(p, beta, backend);
    axpy(real(1), r, p, backend);
    rsold = rsnew;
  }
  return result;
}

} // namespace detail

/// @brief Operator CG for a declared SPD operator.
template<class Op>
  requires SPDLinearOperator<Op, Vector, Vector>
SolverResult cg(const Op& A,
                const Vector& b,
                Vector& x,
                real tol = 1e-10,
                idx max_iter = 1000,
                Backend backend = default_backend) {
  return detail::cg_operator_impl(A, b, x, tol, max_iter, backend);
}

} // namespace num
