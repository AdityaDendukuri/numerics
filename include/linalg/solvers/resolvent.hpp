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

/// A reusable dense LU factorization of (s I - A).  The factorization is
/// intentionally small and dependency-free; callers can solve several RHS
/// columns without rebuilding the shifted matrix.
class ResolventFactor {
public:
  ResolventFactor(cplx s, const Matrix& A) : n_(A.rows()), lu_(n_ * n_) {
    if (A.cols() != n_) { throw std::invalid_argument("ResolventFactor: A must be square");
}
    for (idx i = 0; i < n_; ++i) {
      for (idx j = 0; j < n_; ++j) {
        lu_[(i * n_) + j] = (i == j ? s : cplx(0.0)) - cplx(A(i, j), 0.0);
}
}
    for (idx k = 0; k < n_; ++k) {
      idx pivot = k;
      real best = std::abs(lu_[(k * n_) + k]);
      for (idx i = k + 1; i < n_; ++i) {
        const real v = std::abs(lu_[(i * n_) + k]);
        if (v > best) { best = v; pivot = i; }
      }
      if (best == 0.0) { throw std::runtime_error("ResolventFactor: singular shifted system");
}
      pivots_.push_back(pivot);
      if (pivot != k) {
        for (idx j = 0; j < n_; ++j) { std::swap(lu_[(k * n_) + j], lu_[(pivot * n_) + j]);
}
}
      for (idx i = k + 1; i < n_; ++i) {
        lu_[(i * n_) + k] /= lu_[(k * n_) + k];
        for (idx j = k + 1; j < n_; ++j) {
          lu_[(i * n_) + j] -= lu_[(i * n_) + k] * lu_[(k * n_) + j];
}
      }
    }
  }

  [[nodiscard]] std::vector<cplx> solve(const std::vector<cplx>& b) const {
    if (b.size() != n_) { throw std::invalid_argument("ResolventFactor::solve: dimension mismatch");
}
    std::vector<cplx> x = b;
    for (idx k = 0; k < n_; ++k) {
      if (pivots_[k] != k) { std::swap(x[k], x[pivots_[k]]);
}
}
    for (idx i = 0; i < n_; ++i) {
      for (idx j = 0; j < i; ++j) { x[i] -= lu_[(i * n_) + j] * x[j];
}
}
    for (idx i = n_; i-- > 0;) {
      for (idx j = i + 1; j < n_; ++j) { x[i] -= lu_[(i * n_) + j] * x[j];
}
      x[i] /= lu_[(i * n_) + i];
    }
    return x;
  }

  [[nodiscard]] std::vector<std::vector<cplx>> solve(
      const std::vector<std::vector<cplx>>& rhs) const {
    std::vector<std::vector<cplx>> out;
    out.reserve(rhs.size());
    for (const auto& b : rhs) { out.push_back(solve(b));
}
    return out;
  }

private:
  idx n_;
  std::vector<cplx> lu_;
  std::vector<idx> pivots_;
};

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
      sys[(i * n) + j] = (i == j) ? (s - cplx(A(i, j), 0.0)) : cplx(-A(i, j), 0.0);
    }
  }

  std::vector<cplx> x(n);
  for (idx i = 0; i < n; ++i) {
    x[i] = cplx(b[i], 0.0);
  }

  // LU factorization with partial pivoting for complex dense matrix sys
  std::vector<idx> pivot(n);
  for (idx i = 0; i < n; ++i) { pivot[i] = i;
}

  for (idx k = 0; k < n; ++k) {
    idx max_row = k;
    real max_val = std::abs(sys[(k * n) + k]);
    for (idx i = k + 1; i < n; ++i) {
      real val = std::abs(sys[(i * n) + k]);
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
        std::swap(sys[(k * n) + j], sys[(max_row * n) + j]);
      }
      std::swap(x[k], x[max_row]);
    }

    cplx pivot_val = sys[(k * n) + k];
    for (idx i = k + 1; i < n; ++i) {
      sys[(i * n) + k] /= pivot_val;
      for (idx j = k + 1; j < n; ++j) {
        sys[(i * n) + j] -= sys[(i * n) + k] * sys[(k * n) + j];
      }
    }
  }

  // Forward substitution L y = P b (y in place in x)
  for (idx i = 0; i < n; ++i) {
    for (idx j = 0; j < i; ++j) {
      x[i] -= sys[(i * n) + j] * x[j];
    }
  }

  // Back substitution U x = y
  for (idx i = n; i-- > 0;) {
    for (idx j = i + 1; j < n; ++j) {
      x[i] -= sys[(i * n) + j] * x[j];
    }
    x[i] /= sys[(i * n) + i];
  }

  return x;
}

/// Solve one shifted system for several real RHS vectors, reusing its LU.
inline std::vector<std::vector<cplx>> resolvent_solve_rhs_batch(
    cplx s, const Matrix& A, const std::vector<Vector>& rhs) {
  std::vector<std::vector<cplx>> complex_rhs;
  complex_rhs.reserve(rhs.size());
  for (const auto& b : rhs) {
    std::vector<cplx> cb(b.size());
    for (idx i = 0; i < b.size(); ++i) { cb[i] = cplx(b[i], 0.0);
}
    complex_rhs.push_back(std::move(cb));
  }
  return ResolventFactor(s, A).solve(complex_rhs);
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

/// Batched shifts and RHS columns.  Each shift is factored once, then reused
/// for every RHS column.
inline std::vector<std::vector<std::vector<cplx>>> resolvent_solve_batch(
    const std::vector<cplx>& s_list,
    const Matrix& A,
    const std::vector<Vector>& rhs) {
  std::vector<std::vector<std::vector<cplx>>> results(s_list.size());
#if defined(_OPENMP)
#pragma omp parallel for if(s_list.size() > 4)
#endif
  for (std::size_t k = 0; k < s_list.size(); ++k) {
    results[k] = resolvent_solve_rhs_batch(s_list[k], A, rhs);
}
  return results;
}

} // namespace num
