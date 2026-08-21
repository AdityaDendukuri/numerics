#include "linalg/sparse/umfpack.hpp"
#include <climits>
#include <stdexcept>
#include <utility>

#if defined(NUMERICS_HAS_UMFPACK)
  #include <umfpack.h>
#endif

namespace num {

struct UMFPACKFactor::Impl {
  idx n = 0;
#if defined(NUMERICS_HAS_UMFPACK)
  std::vector<int> ap, ai;
  std::vector<double> ax;
  void* symbolic = nullptr;
  void* numeric = nullptr;
  ~Impl() {
    if (numeric) {
      umfpack_di_free_numeric(&numeric);
    }
    if (symbolic) {
      umfpack_di_free_symbolic(&symbolic);
    }
  }
#endif
};

bool umfpack_available() noexcept {
#if defined(NUMERICS_HAS_UMFPACK)
  return true;
#else
  return false;
#endif
}

UMFPACKFactor::UMFPACKFactor(const SparseMatrix& matrix)
    : impl_(std::make_unique<Impl>()) {
#if defined(NUMERICS_HAS_UMFPACK)
  if (matrix.n_rows() != matrix.n_cols()) {
    throw std::invalid_argument("UMFPACK factorization requires a square matrix");
  }
  if (matrix.n_rows() > INT_MAX || matrix.nnz() > INT_MAX) {
    throw std::overflow_error("UMFPACK int32 interface cannot represent this matrix");
  }
  impl_->n = matrix.n_rows();
  const int n = static_cast<int>(impl_->n);
  impl_->ap.assign(n + 1, 0);
  for (idx row = 0; row < matrix.n_rows(); ++row) {
    for (idx k = matrix.row_ptr()[row]; k < matrix.row_ptr()[row + 1]; ++k) {
      ++impl_->ap[matrix.col_idx()[k] + 1];
    }
  }
  for (int col = 0; col < n; ++col) {
    impl_->ap[col + 1] += impl_->ap[col];
  }
  impl_->ai.resize(matrix.nnz());
  impl_->ax.resize(matrix.nnz());
  std::vector<int> next = impl_->ap;
  for (idx row = 0; row < matrix.n_rows(); ++row) {
    for (idx k = matrix.row_ptr()[row]; k < matrix.row_ptr()[row + 1]; ++k) {
      const int col = static_cast<int>(matrix.col_idx()[k]);
      const int dest = next[col]++;
      impl_->ai[dest] = static_cast<int>(row);
      impl_->ax[dest] = matrix.values()[k];
    }
  }
  double control[UMFPACK_CONTROL], info[UMFPACK_INFO];
  umfpack_di_defaults(control);
  if (umfpack_di_symbolic(n,
                          n,
                          impl_->ap.data(),
                          impl_->ai.data(),
                          impl_->ax.data(),
                          &impl_->symbolic,
                          control,
                          info)
      != UMFPACK_OK) {
    throw std::runtime_error("UMFPACK symbolic analysis failed");
  }
  if (umfpack_di_numeric(impl_->ap.data(),
                         impl_->ai.data(),
                         impl_->ax.data(),
                         impl_->symbolic,
                         &impl_->numeric,
                         control,
                         info)
      != UMFPACK_OK) {
    throw std::runtime_error("UMFPACK numeric factorization failed");
  }
#else
  (void)matrix;
  throw std::runtime_error("Numerics was built without SuiteSparse UMFPACK support");
#endif
}

UMFPACKFactor::~UMFPACKFactor() = default;
UMFPACKFactor::UMFPACKFactor(UMFPACKFactor&&) noexcept = default;
UMFPACKFactor& UMFPACKFactor::operator=(UMFPACKFactor&&) noexcept = default;
idx UMFPACKFactor::size() const noexcept {
  return impl_ ? impl_->n : 0;
}

void UMFPACKFactor::solve(const Vector& rhs, Vector& solution) const {
#if defined(NUMERICS_HAS_UMFPACK)
  if (rhs.size() != impl_->n) {
    throw std::invalid_argument("UMFPACK solve dimension mismatch");
  }
  solution = Vector(impl_->n, 0.0);
  const int status = umfpack_di_solve(UMFPACK_A,
                                      impl_->ap.data(),
                                      impl_->ai.data(),
                                      impl_->ax.data(),
                                      solution.data(),
                                      rhs.data(),
                                      impl_->numeric,
                                      nullptr,
                                      nullptr);
  if (status != UMFPACK_OK) {
    throw std::runtime_error("UMFPACK solve failed");
  }
#else
  (void)rhs;
  (void)solution;
  throw std::runtime_error("Numerics was built without SuiteSparse UMFPACK support");
#endif
}

void UMFPACKFactor::solve(const Matrix& rhs, Matrix& solution) const {
#if defined(NUMERICS_HAS_UMFPACK)
  if (rhs.rows() != impl_->n) {
    throw std::invalid_argument("UMFPACK block solve dimension mismatch");
  }
  solution = Matrix(rhs.rows(), rhs.cols(), 0.0);
  Vector b(impl_->n, 0.0), x(impl_->n, 0.0);
  for (idx col = 0; col < rhs.cols(); ++col) {
    for (idx row = 0; row < rhs.rows(); ++row) {
      b[row] = rhs(row, col);
    }
    solve(b, x);
    for (idx row = 0; row < rhs.rows(); ++row) {
      solution(row, col) = x[row];
    }
  }
#else
  (void)rhs;
  (void)solution;
  throw std::runtime_error("Numerics was built without SuiteSparse UMFPACK support");
#endif
}

} // namespace num
