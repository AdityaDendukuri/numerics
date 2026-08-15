/// @file linalg/solvers/resolvent.hpp
/// @brief Resolvent solver for (s*I - A) x = b for complex s in C using num::cplx.
#pragma once

#include "core/matrix.hpp"
#include "core/vector.hpp"
#include "core/types.hpp"
#include <complex>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <utility>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace num {

/// @brief Solves (s * I - A) x = b for dense A and real b, returning complex solution x.
inline std::vector<cplx> resolvent_solve(cplx s, const Matrix& A, const Vector& b) {
  idx n = A.rows();
  if (A.cols() != n || b.size() != n) {
    throw std::invalid_argument("resolvent_solve: dimension mismatch");
  }

  // Construct (s * I - A) as a complex dense system
  std::vector<cplx> sys(n * n, cplx(0.0, 0.0));
  for (idx i = 0; i < n; ++i) {
    for (idx j = 0; j < n; ++j) {
      sys[i * n + j] = (i == j) ? (s - cplx(A(i, j), 0.0)) : cplx(-A(i, j), 0.0);
    }
  }

  std::vector<cplx> x(n);
  for (idx i = 0; i < n; ++i) {
    x[i] = cplx(b[i], 0.0);
  }

  // LU factorization with partial pivoting for complex dense matrix sys
  std::vector<idx> pivot(n);
  for (idx i = 0; i < n; ++i) pivot[i] = i;

  for (idx k = 0; k < n; ++k) {
    idx max_row = k;
    real max_val = std::abs(sys[k * n + k]);
    for (idx i = k + 1; i < n; ++i) {
      real val = std::abs(sys[i * n + k]);
      if (val > max_val) {
        max_val = val;
        max_row = i;
      }
    }

    if (max_val == 0.0) {
      throw std::runtime_error("resolvent_solve: singular system (s is an eigenvalue of A)");
    }

    if (max_row != k) {
      std::swap(pivot[k], pivot[max_row]);
      for (idx j = 0; j < n; ++j) {
        std::swap(sys[k * n + j], sys[max_row * n + j]);
      }
      std::swap(x[k], x[max_row]);
    }

    cplx pivot_val = sys[k * n + k];
    for (idx i = k + 1; i < n; ++i) {
      sys[i * n + k] /= pivot_val;
      for (idx j = k + 1; j < n; ++j) {
        sys[i * n + j] -= sys[i * n + k] * sys[k * n + j];
      }
    }
  }

  // Forward substitution L y = P b (y in place in x)
  for (idx i = 0; i < n; ++i) {
    for (idx j = 0; j < i; ++j) {
      x[i] -= sys[i * n + j] * x[j];
    }
  }

  // Back substitution U x = y
  for (idx i = n; i-- > 0;) {
    for (idx j = i + 1; j < n; ++j) {
      x[i] -= sys[i * n + j] * x[j];
    }
    x[i] /= sys[i * n + i];
  }

  return x;
}

/// @brief Batched resolvent solver across multiple shift parameters s_k.
inline std::vector<std::vector<cplx>> resolvent_solve_batch(
    const std::vector<cplx>& s_list,
    const Matrix& A,
    const Vector& b
) {
  std::vector<std::vector<cplx>> results(s_list.size());
#if defined(_OPENMP)
#pragma omp parallel for if(s_list.size() > 4)
#endif
  for (std::size_t k = 0; k < s_list.size(); ++k) {
    results[k] = resolvent_solve(s_list[k], A, b);
  }
  return results;
}

} // namespace num
