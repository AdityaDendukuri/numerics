#include "linalg/solvers/krylov.hpp"
#include "kernel/subspace.hpp"
#include "core/vector.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace num {

// Core restarted GMRES implementation (matrix-free)
SolverResult gmres(MatVecFn matvec_fn, idx n, const Vector &b, Vector &x,
                   real tol, idx max_iter, idx restart) {
  if (x.size() != n || b.size() != n)
    throw std::invalid_argument("Dimension mismatch in GMRES");

  restart = std::min(restart, n);
  SolverResult result{0, 0.0, false};

  std::vector<Vector> V;
  V.reserve(restart + 1);
  std::vector<std::vector<real>> H(restart,
                                   std::vector<real>(restart + 1, 0.0));
  std::vector<real> cs(restart, 0.0);
  std::vector<real> sn(restart, 0.0);
  std::vector<real> g(restart + 1, 0.0);

  // Wrap matvec_fn once; scratch buffer reused across all Arnoldi steps
  auto A_op = kernel::subspace::make_op(
      [&](const Vector& in, Vector& out) { matvec_fn(in, out); }, n);
  Vector scratch(n);

  idx total_iters = 0;

  while (total_iters < max_iter) {
    Vector r(n);
    matvec_fn(x, r);
    for (idx i = 0; i < n; ++i)
      r[i] = b[i] - r[i];

    real beta = 0.0;
    for (idx i = 0; i < n; ++i)
      beta += r[i] * r[i];
    beta = std::sqrt(beta);

    result.residual = beta;
    if (beta < tol) {
      result.converged = true;
      break;
    }

    V.clear();
    V.emplace_back(n);
    for (idx i = 0; i < n; ++i)
      V[0][i] = r[i] / beta;

    for (auto &col : H)
      std::fill(col.begin(), col.end(), 0.0);
    std::fill(cs.begin(), cs.end(), 0.0);
    std::fill(sn.begin(), sn.end(), 0.0);
    std::fill(g.begin(), g.end(), 0.0);
    g[0] = beta;

    idx j = 0;
    for (; j < restart && total_iters < max_iter; ++j, ++total_iters) {
      result.iterations = total_iters + 1;

      const real h_next = kernel::subspace::arnoldi_step(
          A_op, V, H[j], j, scratch, real(1e-15));

      if (h_next < 1e-15) {
        ++j;
        break;
      }

      for (idx i = 0; i < j; ++i) {
        real tmp = cs[i] * H[j][i] + sn[i] * H[j][i + 1];
        H[j][i + 1] = -sn[i] * H[j][i] + cs[i] * H[j][i + 1];
        H[j][i] = tmp;
      }

      real h0 = H[j][j], h1 = H[j][j + 1];
      real denom = std::sqrt(h0 * h0 + h1 * h1);
      if (denom < 1e-15) {
        cs[j] = 1.0;
        sn[j] = 0.0;
      } else {
        cs[j] = h0 / denom;
        sn[j] = h1 / denom;
      }

      H[j][j] = cs[j] * h0 + sn[j] * h1;
      H[j][j + 1] = 0.0;

      g[j + 1] = -sn[j] * g[j];
      g[j] = cs[j] * g[j];

      result.residual = std::abs(g[j + 1]);
      if (result.residual < tol) {
        result.converged = true;
        ++j;
        break;
      }
    }

    idx m = j;
    std::vector<real> y(m, 0.0);
    for (idx i = m; i > 0;) {
      --i;
      y[i] = g[i];
      for (idx k = i + 1; k < m; ++k)
        y[i] -= H[k][i] * y[k];
      y[i] /= H[i][i];
    }

    for (idx i = 0; i < m; ++i)
      for (idx k = 0; k < n; ++k)
        x[k] += y[i] * V[i][k];

    if (result.converged)
      break;
  }

  return result;
}

// Sparse overload
SolverResult gmres(const SparseMatrix &A, const Vector &b, Vector &x, real tol,
                   idx max_iter, idx restart) {
  if (A.n_rows() != A.n_cols())
    throw std::invalid_argument("GMRES requires a square matrix");
  idx n = A.n_rows();
  MatVecFn mv = [&](const Vector &in, Vector &out) {
    out = Vector(n);
    sparse_matvec(A, in, out);
  };
  return gmres(mv, n, b, x, tol, max_iter, restart);
}

// Dense overload  -- wraps the matrix-free core with a backend-parameterized
// matvec
SolverResult gmres(const Matrix &A, const Vector &b, Vector &x, real tol,
                   idx max_iter, idx restart, Backend backend) {
  if (A.rows() != A.cols())
    throw std::invalid_argument("GMRES requires a square matrix");
  idx n = A.rows();
  MatVecFn mv = [&](const Vector &in, Vector &out) {
    out = Vector(n);
    matvec(A, in, out, backend);
  };
  return gmres(mv, n, b, x, tol, max_iter, restart);
}

} // namespace num
