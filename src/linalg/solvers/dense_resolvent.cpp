#include "linalg/solvers/dense_resolvent.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#if defined(NUMERICS_HAS_LAPACK)
  #include "core/parallel/lapack_wrapper.hpp"
#endif

namespace num {

struct DenseResolventSolver::Impl {
  explicit Impl(Matrix input)
      : matrix(std::move(input)),
        lu(matrix.size()) {
    if (matrix.rows() != matrix.cols()) {
      throw std::invalid_argument("DenseResolventSolver requires a square matrix");
    }
#if defined(NUMERICS_HAS_LAPACK)
    pivots.resize(matrix.rows());
#else
    pivots.resize(matrix.rows());
#endif
  }

  Matrix matrix;
  std::vector<cplx> lu;
#if defined(NUMERICS_HAS_LAPACK)
  std::vector<lapack_int> pivots;
#else
  std::vector<idx> pivots;
#endif
  bool factored = false;
};

namespace {

Matrix dense_copy(const SparseMatrix& sparse) {
  Matrix dense(sparse.n_rows(), sparse.n_cols(), 0.0);
  for (idx row = 0; row < sparse.n_rows(); ++row) {
    for (idx entry = sparse.row_ptr()[row]; entry < sparse.row_ptr()[row + 1]; ++entry) {
      dense(row, sparse.col_idx()[entry]) = sparse.values()[entry];
    }
  }
  return dense;
}

} // namespace

DenseResolventSolver::DenseResolventSolver(const Matrix& matrix)
    : impl_(std::make_unique<Impl>(matrix)) {}

DenseResolventSolver::DenseResolventSolver(const SparseMatrix& matrix)
    : impl_(std::make_unique<Impl>(dense_copy(matrix))) {}

DenseResolventSolver::~DenseResolventSolver() = default;
DenseResolventSolver::DenseResolventSolver(DenseResolventSolver&&) noexcept = default;
DenseResolventSolver&
DenseResolventSolver::operator=(DenseResolventSolver&&) noexcept = default;

idx DenseResolventSolver::size() const noexcept {
  return impl_ ? impl_->matrix.rows() : 0;
}

void DenseResolventSolver::factorize(cplx shift) {
  const idx n = impl_->matrix.rows();
  for (idx row = 0; row < n; ++row) {
    for (idx column = 0; column < n; ++column) {
      impl_->lu[(row * n) + column] =
          (row == column ? shift : cplx(0.0, 0.0)) - impl_->matrix(row, column);
    }
  }

#if defined(NUMERICS_HAS_LAPACK)
  int info = 0;
  const auto lapack_n = static_cast<lapack_int>(n);
  #if defined(NUMERICS_LAPACK_ACCELERATE)
  zgetrf_(&lapack_n, &lapack_n, impl_->lu.data(), &lapack_n, impl_->pivots.data(), &info);
  #else
  info = LAPACKE_zgetrf(LAPACK_ROW_MAJOR,
                        lapack_n,
                        lapack_n,
                        reinterpret_cast<lapack_complex_double*>(impl_->lu.data()),
                        lapack_n,
                        impl_->pivots.data());
  #endif
  if (info != 0) {
    throw std::runtime_error("DenseResolventSolver LU factorization failed");
  }
#else
  for (idx column = 0; column < n; ++column) {
    idx pivot = column;
    double largest = std::abs(impl_->lu[(column * n) + column]);
    for (idx row = column + 1; row < n; ++row) {
      const double candidate = std::abs(impl_->lu[(row * n) + column]);
      if (candidate > largest) {
        largest = candidate;
        pivot = row;
      }
    }
    if (largest == 0.0) {
      throw std::runtime_error("DenseResolventSolver encountered a singular matrix");
    }
    impl_->pivots[column] = pivot;
    if (pivot != column) {
      for (idx entry = 0; entry < n; ++entry) {
        std::swap(impl_->lu[(column * n) + entry], impl_->lu[(pivot * n) + entry]);
      }
    }
    for (idx row = column + 1; row < n; ++row) {
      impl_->lu[(row * n) + column] /= impl_->lu[(column * n) + column];
      for (idx entry = column + 1; entry < n; ++entry) {
        impl_->lu[(row * n) + entry] -=
            impl_->lu[(row * n) + column] * impl_->lu[(column * n) + entry];
      }
    }
  }
#endif
  impl_->factored = true;
}

std::vector<cplx>
DenseResolventSolver::solve(const std::vector<cplx>& rhs) const {
  std::vector<cplx> result;
  solve(rhs, result);
  return result;
}

void DenseResolventSolver::solve(const std::vector<cplx>& rhs,
                                 std::vector<cplx>& result) const {
  const idx n = impl_->matrix.rows();
  if (!impl_->factored || rhs.size() != n) {
    throw std::invalid_argument(
        "DenseResolventSolver: factorization or matching right-hand side required");
  }
  result = rhs;

#if defined(NUMERICS_HAS_LAPACK)
  int info = 0;
  const auto lapack_n = static_cast<lapack_int>(n);
  constexpr lapack_int one = 1;
  #if defined(NUMERICS_LAPACK_ACCELERATE)
  // Row-major storage is interpreted by Fortran as the transpose.
  constexpr char transpose = 'T';
  zgetrs_(&transpose,
          &lapack_n,
          &one,
          impl_->lu.data(),
          &lapack_n,
          impl_->pivots.data(),
          result.data(),
          &lapack_n,
          &info);
  #else
  info = LAPACKE_zgetrs(LAPACK_ROW_MAJOR,
                        'N',
                        lapack_n,
                        one,
                        reinterpret_cast<const lapack_complex_double*>(impl_->lu.data()),
                        lapack_n,
                        impl_->pivots.data(),
                        reinterpret_cast<lapack_complex_double*>(result.data()),
                        one);
  #endif
  if (info != 0) {
    throw std::runtime_error("DenseResolventSolver solve failed");
  }
#else
  for (idx column = 0; column < n; ++column) {
    if (impl_->pivots[column] != column) {
      std::swap(result[column], result[impl_->pivots[column]]);
    }
  }
  for (idx row = 0; row < n; ++row) {
    for (idx column = 0; column < row; ++column) {
      result[row] -= impl_->lu[(row * n) + column] * result[column];
    }
  }
  for (idx row = n; row-- > 0;) {
    for (idx column = row + 1; column < n; ++column) {
      result[row] -= impl_->lu[(row * n) + column] * result[column];
    }
    result[row] /= impl_->lu[(row * n) + row];
  }
#endif
}

} // namespace num
