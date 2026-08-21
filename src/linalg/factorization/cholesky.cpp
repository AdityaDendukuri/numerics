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
  int info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR,
                            'L',
                            static_cast<lapack_int>(n),
                            L.data(),
                            static_cast<lapack_int>(n));
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

  x = b;
#if defined(NUMERICS_HAS_LAPACK)
  const int info = LAPACKE_dpotrs(LAPACK_ROW_MAJOR,
                                  'L',
                                  static_cast<lapack_int>(n),
                                  1,
                                  f.L.data(),
                                  static_cast<lapack_int>(n),
                                  x.data(),
                                  1);
  if (info != 0) {
    throw std::runtime_error("cholesky_solve: LAPACK solve failed");
  }
#else
  for (idx i = 0; i < n; ++i) {
    real sum = x[i];
    for (idx k = 0; k < i; ++k) {
      sum -= f.L(i, k) * x[k];
    }
    x[i] = sum / f.L(i, i);
  }

  for (idx ii = n; ii > 0;) {
    --ii;
    real sum = x[ii];
    for (idx k = ii + 1; k < n; ++k) {
      sum -= f.L(k, ii) * x[k];
    }
    x[ii] = sum / f.L(ii, ii);
  }
#endif
}

void cholesky_solve(const CholeskyResult& f, const Matrix& B, Matrix& X) {
  if (!f.success) {
    throw std::invalid_argument("cholesky_solve: factorization failed");
  }
  const idx n = f.L.rows();
  if (f.L.cols() != n || B.rows() != n) {
    throw std::invalid_argument("cholesky_solve: dimension mismatch");
  }

  X = B;
#if defined(NUMERICS_HAS_LAPACK)
  const int info = LAPACKE_dpotrs(LAPACK_ROW_MAJOR,
                                  'L',
                                  static_cast<lapack_int>(n),
                                  static_cast<lapack_int>(B.cols()),
                                  f.L.data(),
                                  static_cast<lapack_int>(n),
                                  X.data(),
                                  static_cast<lapack_int>(B.cols()));
  if (info != 0) {
    throw std::runtime_error("cholesky_solve: LAPACK block solve failed");
  }
#else
  for (idx row = 0; row < n; ++row) {
    for (idx column = 0; column < B.cols(); ++column) {
      real value = X(row, column);
      for (idx k = 0; k < row; ++k) {
        value -= f.L(row, k) * X(k, column);
      }
      X(row, column) = value / f.L(row, row);
    }
  }
  for (idx row = n; row-- > 0;) {
    for (idx column = 0; column < B.cols(); ++column) {
      real value = X(row, column);
      for (idx k = row + 1; k < n; ++k) {
        value -= f.L(k, row) * X(k, column);
      }
      X(row, column) = value / f.L(row, row);
    }
  }
#endif
}

void solve_in_place(const CholeskyResult& f, Vector& right_hand_side) {
  Vector result(right_hand_side.size(), 0.0);
  cholesky_solve(f, right_hand_side, result);
  right_hand_side = std::move(result);
}

void solve_in_place(const CholeskyResult& f, Matrix& right_hand_sides) {
  Matrix result;
  cholesky_solve(f, right_hand_sides, result);
  right_hand_sides = std::move(result);
}

void cholesky_update(CholeskyResult& factor, const Vector& update) {
  if (!factor.success || factor.L.rows() != update.size()) {
    throw std::invalid_argument("cholesky_update: invalid factor or update size");
  }
  Vector work = update;
  for (idx column = 0; column < work.size(); ++column) {
    const real diagonal = factor.L(column, column);
    const real replacement = std::hypot(diagonal, work[column]);
    const real cosine = replacement / diagonal;
    const real sine = work[column] / diagonal;
    factor.L(column, column) = replacement;
    for (idx row = column + 1; row < work.size(); ++row) {
      factor.L(row, column) = (factor.L(row, column) + (sine * work[row])) / cosine;
      work[row] = (cosine * work[row]) - (sine * factor.L(row, column));
    }
  }
}

void cholesky_downdate(CholeskyResult& factor, const Vector& update) {
  if (!factor.success || factor.L.rows() != update.size()) {
    throw std::invalid_argument("cholesky_downdate: invalid factor or update size");
  }
  Matrix candidate = factor.L;
  Vector work = update;
  for (idx column = 0; column < work.size(); ++column) {
    const real diagonal = candidate(column, column);
    const real square = (diagonal * diagonal) - (work[column] * work[column]);
    if (!(square > 0.0)) {
      throw std::domain_error("cholesky_downdate: result is not positive definite");
    }
    const real replacement = std::sqrt(square);
    const real cosine = replacement / diagonal;
    const real sine = work[column] / diagonal;
    candidate(column, column) = replacement;
    for (idx row = column + 1; row < work.size(); ++row) {
      candidate(row, column) = (candidate(row, column) - (sine * work[row])) / cosine;
      work[row] = (cosine * work[row]) - (sine * candidate(row, column));
    }
  }
  factor.L = std::move(candidate);
}

} // namespace num
