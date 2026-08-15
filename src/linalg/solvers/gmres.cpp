#include "linalg/solvers/gmres.hpp"
#include "linalg/sparse/sparse_op.hpp"
#include "operator/dense.hpp"
#include <stdexcept>

namespace num {

// Sparse overload
SolverResult gmres(const SparseMatrix& A,
                   const Vector& b,
                   Vector& x,
                   real tol,
                   idx max_iter,
                   idx restart) {
  if (A.n_rows() != A.n_cols())
    throw std::invalid_argument("GMRES requires a square matrix");
  operators::SparseOp op(A);
  return gmres(op, b, x, tol, max_iter, restart);
}

// Dense overload  -- wraps the matrix-free core with a backend-parameterized
// matvec
SolverResult gmres(const Matrix& A,
                   const Vector& b,
                   Vector& x,
                   real tol,
                   idx max_iter,
                   idx restart,
                   Backend backend) {
  if (A.rows() != A.cols())
    throw std::invalid_argument("GMRES requires a square matrix");
  operators::DenseOp op(A, backend);
  return gmres(op, b, x, tol, max_iter, restart);
}

} // namespace num
