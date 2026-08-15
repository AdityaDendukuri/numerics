/// @file expv.hpp
/// @brief Krylov subspace matrix exponential-vector product: compute exp(t*A)*v
///
/// Approximates \f$\exp(tA)v \approx \|v\| Q_m \exp(tH_m)e_1\f$ where
/// \f$AQ_m \approx Q_{m+1}\bar{H}_m\f$ is the Arnoldi relation.
/// @todo Add adaptive step subdivision and an a posteriori error estimate for
/// large \f$|t|\|A\|\f$.
#pragma once

#include "core/matrix.hpp"
#include "core/types.hpp"
#include "core/vector.hpp"
#include "kernel/subspace.hpp"
#include "linalg/sparse/sparse.hpp"
#include "core/concepts.hpp"
#include <stdexcept>
#include <utility>
#include <vector>

namespace num {

namespace detail {
Matrix dense_expm_pade6(const Matrix& A);
}

/// @brief Compute \f$\exp(tA)v\f$ for any \f$y=A x\f$ adapter.
template<class Op>
  requires LinearOperator<Op, Vector, Vector>
Vector expv(real t, const Op& A, const Vector& v, int m_max = 30, real tol = 1e-8) {
  const idx n = A.rows();
  if (A.cols() != n || v.size() != n) {
    throw std::invalid_argument("expv: dimension mismatch");
  }

  real beta = norm(v);
  if (beta < 1e-300) {
    return Vector(n, 0.0);
  }

  std::vector<Vector> V;
  V.reserve(m_max + 1);

  Vector v0(n);
  for (idx i = 0; i < n; i++) {
    v0[i] = v[i] / beta;
  }
  V.push_back(std::move(v0));

  Matrix H(m_max + 1, m_max, 0.0);
  int m_actual = m_max;
  std::vector<real> h_col(m_max + 1, 0.0);

  for (int j = 0; j < m_max; j++) {
    Vector w(n, 0.0);
    A.apply(V[j], w);

    const real h_next = kernel::subspace::mgs_orthogonalize(V, w, h_col, j + 1);
    for (int i = 0; i <= j; i++) {
      H(i, j) = h_col[i];
    }
    H(j + 1, j) = h_next;

    if (h_next < tol) {
      m_actual = j + 1;
      break;
    }

    scale(w, real(1) / h_next);
    V.push_back(std::move(w));
  }

  Matrix Hm(m_actual, m_actual, 0.0);
  for (int i = 0; i < m_actual; i++) {
    for (int j = 0; j < m_actual; j++) {
      Hm(i, j) = t * H(i, j);
    }
  }

  Matrix E = detail::dense_expm_pade6(Hm);

  Vector result(n, 0.0);
  for (int j = 0; j < m_actual; j++) {
    axpy(beta * E(j, 0), V[j], result);
  }

  return result;
}

Vector expv(real t,
            const SparseMatrix& A,
            const Vector& v,
            int m_max = 30,
            real tol = 1e-8);

} // namespace num
