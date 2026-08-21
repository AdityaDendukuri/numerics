#include "linalg/solvers/sparse_resolvent.hpp"
#include <algorithm>
#include <climits>
#include <stdexcept>
#include <utility>

#if defined(NUMERICS_HAS_UMFPACK)
  #include <umfpack.h>
#endif

namespace num {

bool sparse_resolvent_available() noexcept {
#if defined(NUMERICS_HAS_UMFPACK)
  return true;
#else
  return false;
#endif
}

struct SparseResolventSolver::Impl {
  idx n = 0;
#if defined(NUMERICS_HAS_UMFPACK)
  std::vector<int> ap, ai;
  std::vector<double> ar, az, xr, xz, br, bz;
  std::vector<int> diagonal;
  cplx current_shift = {0.0, 0.0};
  void* symbolic = nullptr;
  void* numeric = nullptr;
  ~Impl() {
    if (numeric) {
      umfpack_zi_free_numeric(&numeric);
    }
    if (symbolic) {
      umfpack_zi_free_symbolic(&symbolic);
    }
  }
#endif
};

SparseResolventSolver::SparseResolventSolver(const SparseMatrix& A,
                                             SparseResolventOptions options)
    : impl_(std::make_unique<Impl>()) {
  if (A.n_rows() != A.n_cols()) {
    throw std::invalid_argument("SparseResolventSolver requires a square matrix");
  }
  if (A.n_rows() > INT_MAX || A.nnz() > INT_MAX) {
    throw std::overflow_error("SparseResolventSolver int32 interface overflow");
  }
  impl_->n = A.n_rows();
#if defined(NUMERICS_HAS_UMFPACK)
  const int n = static_cast<int>(impl_->n);
  impl_->ap.assign(n + 1, 0);
  for (idx i = 0; i < impl_->n; ++i) {
    for (idx k = A.row_ptr()[i]; k < A.row_ptr()[i + 1]; ++k) {
      ++impl_->ap[A.col_idx()[k] + 1];
    }
  }
  for (int j = 0; j < n; ++j) {
    impl_->ap[j + 1] += impl_->ap[j];
  }
  impl_->ai.resize(A.nnz());
  impl_->ar.resize(A.nnz());
  std::vector<int> next = impl_->ap;
  for (idx i = 0; i < impl_->n; ++i) {
    for (idx k = A.row_ptr()[i]; k < A.row_ptr()[i + 1]; ++k) {
      const int p = next[A.col_idx()[k]]++;
      impl_->ai[p] = static_cast<int>(i);
      // Store -A so adding the complex shift to the diagonal forms sI - A.
      impl_->ar[p] = -A.values()[k];
    }
  }
  impl_->diagonal.assign(A.n_cols(), -1);
  for (int col = 0; col < n; ++col) {
    const int begin = impl_->ap[col];
    const int end = impl_->ap[col + 1];
    std::vector<std::pair<int, double>> entries;
    entries.reserve(static_cast<std::size_t>(end - begin));
    for (int p = begin; p < end; ++p) {
      entries.emplace_back(impl_->ai[p], impl_->ar[p]);
    }
    std::sort(entries.begin(), entries.end(), [](const auto& lhs, const auto& rhs) {
      return lhs.first < rhs.first;
    });
    for (int offset = 0; offset < end - begin; ++offset) {
      impl_->ai[begin + offset] = entries[offset].first;
      impl_->ar[begin + offset] = entries[offset].second;
      if (entries[offset].first == col) {
        impl_->diagonal[col] = begin + offset;
      }
    }
  }
  impl_->az.assign(A.nnz(), 0.0);
  impl_->xr.resize(impl_->n);
  impl_->xz.resize(impl_->n);
  impl_->br.resize(impl_->n);
  impl_->bz.resize(impl_->n);
  double control[UMFPACK_CONTROL], info[UMFPACK_INFO];
  umfpack_zi_defaults(control);
  if (options.symmetric_pattern) {
    control[UMFPACK_STRATEGY] = UMFPACK_STRATEGY_SYMMETRIC;
  }
  if (umfpack_zi_symbolic(n,
                          n,
                          impl_->ap.data(),
                          impl_->ai.data(),
                          impl_->ar.data(),
                          impl_->az.data(),
                          &impl_->symbolic,
                          control,
                          info)
      != UMFPACK_OK) {
    throw std::runtime_error("UMFPACK complex symbolic analysis failed");
  }
#else
  (void)A;
#endif
}

SparseResolventSolver::~SparseResolventSolver() = default;
SparseResolventSolver::SparseResolventSolver(SparseResolventSolver&&) noexcept = default;
SparseResolventSolver& SparseResolventSolver::operator=(
  SparseResolventSolver&&) noexcept = default;
idx SparseResolventSolver::size() const noexcept {
  return impl_ ? impl_->n : 0;
}

void SparseResolventSolver::factorize(cplx shift) {
#if defined(NUMERICS_HAS_UMFPACK)
  if (impl_->numeric) {
    umfpack_zi_free_numeric(&impl_->numeric);
  }
  const double delta_real = shift.real() - impl_->current_shift.real();
  for (idx j = 0; j < impl_->n; ++j) {
    const int p = impl_->diagonal[j];
    if (p < 0) {
      throw std::runtime_error("SparseResolventSolver requires explicit diagonal");
    }
    impl_->ar[p] += delta_real;
    impl_->az[p] = shift.imag();
  }
  impl_->current_shift = shift;
  double control[UMFPACK_CONTROL], info[UMFPACK_INFO];
  umfpack_zi_defaults(control);
  if (umfpack_zi_numeric(impl_->ap.data(),
                         impl_->ai.data(),
                         impl_->ar.data(),
                         impl_->az.data(),
                         impl_->symbolic,
                         &impl_->numeric,
                         control,
                         info)
      != UMFPACK_OK) {
    throw std::runtime_error("UMFPACK complex numeric factorization failed");
  }
#else
  (void)shift;
  throw std::runtime_error(
    "SparseResolventSolver requires SuiteSparse UMFPACK complex support");
#endif
}

std::vector<cplx> SparseResolventSolver::solve(const std::vector<cplx>& rhs) const {
  std::vector<cplx> out;
  solve(rhs, out);
  return out;
}

void SparseResolventSolver::solve(const std::vector<cplx>& rhs,
                                  std::vector<cplx>& out) const {
#if defined(NUMERICS_HAS_UMFPACK)
  if (!impl_->numeric || rhs.size() != impl_->n) {
    throw std::invalid_argument(
      "SparseResolventSolver: factorization or dimension missing");
  }
  for (idx i = 0; i < impl_->n; ++i) {
    impl_->br[i] = rhs[i].real(), impl_->bz[i] = rhs[i].imag();
  }
  if (umfpack_zi_solve(UMFPACK_A,
                       impl_->ap.data(),
                       impl_->ai.data(),
                       impl_->ar.data(),
                       impl_->az.data(),
                       impl_->xr.data(),
                       impl_->xz.data(),
                       impl_->br.data(),
                       impl_->bz.data(),
                       impl_->numeric,
                       nullptr,
                       nullptr)
      != UMFPACK_OK) {
    throw std::runtime_error("UMFPACK complex solve failed");
  }
  out.resize(impl_->n);
  for (idx i = 0; i < impl_->n; ++i) {
    out[i] = {impl_->xr[i], impl_->xz[i]};
  }
#else
  (void)rhs;
  (void)out;
  throw std::runtime_error(
    "SparseResolventSolver requires SuiteSparse UMFPACK complex support");
#endif
}

std::vector<std::vector<cplx>> SparseResolventSolver::solve(
  const std::vector<std::vector<cplx>>& rhs) const {
  std::vector<std::vector<cplx>> out;
  out.reserve(rhs.size());
  for (const auto& b : rhs) {
    out.push_back(solve(b));
  }
  return out;
}

} // namespace num
