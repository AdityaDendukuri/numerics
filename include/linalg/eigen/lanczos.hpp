/// @file eigen/lanczos.hpp
/// @brief Lanczos eigensolver for symmetric operators.
///
/// Builds an orthonormal basis \f$Q_m\f$ such that
/// \f$Q_m^T A Q_m = T_m\f$, with \f$T_m\f$ tridiagonal.
/// @todo Add thick-restart Lanczos and selective reorthogonalization controls.
#pragma once

#include "core/matrix.hpp"
#include "core/policy.hpp"
#include "core/vector.hpp"
#include "kernel/subspace.hpp"
#include "linalg/eigen/jacobi_eig.hpp"
#include "linalg/sparse/sparse.hpp"
#include "operator/concepts.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace num {

/// Largest Ritz pairs and residual-based convergence metadata.
struct LanczosResult {
  Vector ritz_values; ///< Requested Ritz values in ascending order.
  Matrix ritz_vectors; ///< Ritz vectors stored as columns.
  idx steps = 0; ///< Lanczos basis vectors generated.
  bool converged = false; ///< Whether all returned Ritz pairs met tolerance.
};

namespace detail {

template<class Op>
  requires LinearOperator<Op, Vector, Vector>
LanczosResult lanczos_operator_impl(const Op& A,
                                    idx k,
                                    real tol,
                                    idx max_steps,
                                    Backend backend) {
  (void)backend;
  const idx n = A.rows();
  if (A.cols() != n) {
    throw std::invalid_argument("lanczos: operator must be square");
  }
  if (k == 0 || k > n) {
    throw std::invalid_argument("lanczos: k must satisfy 0 < k <= n");
  }

  if (max_steps == 0) {
    max_steps = std::min(3 * k, n);
  }
  max_steps = std::min(max_steps, n);

  Matrix V(n, max_steps, 0.0);
  Vector alpha(max_steps, 0.0);
  Vector beta(max_steps, 0.0);

  for (idx i = 0; i < n; ++i) {
    V(i, 0) = (i == 0) ? 1.0 : 0.0;
  }

  idx steps = 0;

  for (idx j = 0; j < max_steps; ++j) {
    Vector vj(n);
    for (idx i = 0; i < n; ++i) {
      vj[i] = V(i, j);
    }

    Vector w(n, 0.0);
    A.apply(vj, w);

    const real a = dot(vj, w);
    alpha[j] = a;

    axpy(-a, vj, w);
    if (j > 0) {
      for (idx i = 0; i < n; ++i) {
        w[i] -= beta[j - 1] * V(i, j - 1);
      }
    }

    const real b = kernel::subspace::mgs_orthogonalize(V, j + 1, w);
    ++steps;

    if (b < real(1e-12)) {
      break;
    }

    beta[j] = b;

    if (j + 1 < max_steps) {
      for (idx i = 0; i < n; ++i) {
        V(i, j + 1) = w[i] / b;
      }
    }
  }

  const idx m = steps;
  Matrix T(m, m, 0.0);
  for (idx j = 0; j < m; ++j) {
    T(j, j) = alpha[j];
    if (j + 1 < m) {
      T(j, j + 1) = beta[j];
      T(j + 1, j) = beta[j];
    }
  }

  EigenResult teig = eig_sym(T, tol * real(1e-2));
  const idx nret = std::min(k, m);

  Matrix ritz_vecs(n, nret, 0.0);
  for (idx i = 0; i < nret; ++i) {
    const idx ti = m - nret + i;
    for (idx j = 0; j < m; ++j) {
      const real coeff = teig.vectors(j, ti);
      for (idx r = 0; r < n; ++r) {
        ritz_vecs(r, i) += coeff * V(r, j);
      }
    }
  }

  Vector ritz_vals(nret);
  for (idx i = 0; i < nret; ++i) {
    ritz_vals[i] = teig.values[m - nret + i];
  }

  bool all_converged = true;
  for (idx i = 0; i < nret; ++i) {
    Vector u(n);
    for (idx r = 0; r < n; ++r) {
      u[r] = ritz_vecs(r, i);
    }

    Vector Au(n, 0.0);
    A.apply(u, Au);

    real res = 0;
    const real lam = ritz_vals[i];
    for (idx r = 0; r < n; ++r) {
      const real d = Au[r] - (lam * u[r]);
      res += d * d;
    }
    if (std::sqrt(res) > tol) {
      all_converged = false;
      break;
    }
  }

  return {ritz_vals, ritz_vecs, steps, all_converged};
}

} // namespace detail

/// @brief Operator Lanczos for a declared symmetric \f$y=A x\f$ adapter.
template<class Op>
  requires SymmetricLinearOperator<Op, Vector, Vector>
LanczosResult lanczos(const Op& A,
                      idx k,
                      real tol = 1e-10,
                      idx max_steps = 0,
                      Backend backend = Backend::seq) {
  return detail::lanczos_operator_impl(A, k, tol, max_steps, backend);
}

/// Compute the largest k Ritz pairs of a stored symmetric dense matrix.
LanczosResult lanczos(const Matrix& A,
                      idx k,
                      real tol = 1e-10,
                      idx max_steps = 0,
                      Backend backend = Backend::seq);

/// Compute the largest k Ritz pairs of a stored symmetric sparse matrix.
LanczosResult lanczos(const SparseMatrix& A,
                      idx k,
                      real tol = 1e-10,
                      idx max_steps = 0,
                      Backend backend = Backend::seq);

} // namespace num
