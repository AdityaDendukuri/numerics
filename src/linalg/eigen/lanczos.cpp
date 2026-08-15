/// @file eigen/lanczos.cpp
/// @brief Lanczos overload wrappers.

#include "linalg/eigen/lanczos.hpp"
#include "linalg/sparse/sparse_op.hpp"
#include "operator/dense.hpp"
#include "operator/properties.hpp"
#include <stdexcept>

namespace num {

LanczosResult lanczos(const Matrix& A, idx k, real tol, idx max_steps, Backend backend) {
  if (A.rows() != A.cols()) {
    throw std::invalid_argument("lanczos: matrix must be square");
  }
  operators::DenseOp op(A, backend);
  return lanczos(operators::assume_symmetric(op), k, tol, max_steps, backend);
}

LanczosResult lanczos(const SparseMatrix& A,
                      idx k,
                      real tol,
                      idx max_steps,
                      Backend backend) {
  if (A.n_rows() != A.n_cols()) {
    throw std::invalid_argument("lanczos: matrix must be square");
  }
  (void)backend;
  operators::SparseOp op(A);
  return lanczos(operators::assume_symmetric(op), k, tol, max_steps, backend);
}

} // namespace num
