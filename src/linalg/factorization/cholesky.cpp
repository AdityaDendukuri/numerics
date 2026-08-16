#include "linalg/factorization/cholesky.hpp"
#include <cmath>
#include <stdexcept>
#include <vector>

#if defined(NUMERICS_HAS_LAPACK)
  #include "core/parallel/lapack_wrapper.hpp"
#endif

namespace num {

CholeskyResult cholesky(const linalg::SPDMatrix<Matrix>& A) {
  return cholesky(A.base());
}

CholeskyResult cholesky(const Matrix& A) {
  if (A.rows() != A.cols()) {
    throw std::invalid_argument("cholesky: matrix must be square");
  }

  const idx n = A.rows();

#if defined(NUMERICS_HAS_LAPACK)
  Matrix L = A;
  int info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', static_cast<lapack_int>(n), L.data(), static_cast<lapack_int>(n));
  if (info != 0) {
    return {std::move(L), false};
  }

  // Zero out upper triangle for lower triangular result L
  for (idx i = 0; i < n; ++i) {
    for (idx j = i + 1; j < n; ++j) {
      L(i, j) = 0.0;
    }
  }

  return {std::move(L), true};
#else
  Matrix L(n, n, 0.0);

  for (idx i = 0; i < n; ++i) {
    for (idx j = 0; j <= i; ++j) {
      real sum = A(i, j);
      for (idx k = 0; k < j; ++k) {
        sum -= L(i, k) * L(j, k);
      }

      if (i == j) {
        if (sum <= real(0)) {
          return {std::move(L), false};
        }
        L(i, j) = std::sqrt(sum);
      } else {
        L(i, j) = sum / L(j, j);
      }
    }
  }

  return {std::move(L), true};
#endif
}

void cholesky_solve(const CholeskyResult& f, const Vector& b, Vector& x) {
  if (!f.success) {
    throw std::invalid_argument("cholesky_solve: factorization failed");
  }
  const idx n = f.L.rows();
  if (f.L.cols() != n || b.size() != n || x.size() != n) {
    throw std::invalid_argument("cholesky_solve: dimension mismatch");
  }

  Vector y(n, 0.0);
  for (idx i = 0; i < n; ++i) {
    real sum = b[i];
    for (idx k = 0; k < i; ++k) {
      sum -= f.L(i, k) * y[k];
    }
    y[i] = sum / f.L(i, i);
  }

  for (idx ii = n; ii > 0;) {
    --ii;
    real sum = y[ii];
    for (idx k = ii + 1; k < n; ++k) {
      sum -= f.L(k, ii) * x[k];
    }
    x[ii] = sum / f.L(ii, ii);
  }
}

} // namespace num
