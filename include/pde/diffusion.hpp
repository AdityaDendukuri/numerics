/// @file pde/diffusion.hpp
/// @brief Diffusion operators and implicit system builders for 2D grids.
/// @todo Add structured SPD operators for periodic diffusion, 3D grids, and
/// matrix-free stencil application.
#pragma once

#include "core/policy.hpp"
#include "core/vector.hpp"
#include "fields/grid2d.hpp"
#include "linalg/solvers/cg.hpp"
#include "linalg/solvers/linear_solver.hpp"
#include "linalg/sparse/sparse.hpp"
#include "linalg/sparse/sparse_op.hpp"
#include "operator/properties.hpp"
#include "pde/stencil.hpp"

namespace num::pde {

inline SparseMatrix backward_euler_matrix(int N, double coeff);

class BackwardEulerOperator2D final {
public:
  using symmetric_operator_tag = void;
  using spd_operator_tag = void;

  BackwardEulerOperator2D(int N, double coeff)
      : A_(backward_euler_matrix(N, coeff)) {}

  void apply(const Vector& x, Vector& y) const { sparse_matvec(A_, x, y); }
  [[nodiscard]] idx rows() const noexcept { return A_.n_rows(); }
  [[nodiscard]] idx cols() const noexcept { return A_.n_cols(); }
  [[nodiscard]] const SparseMatrix& matrix() const noexcept { return A_; }

private:
  SparseMatrix A_;
};

inline void diffusion_step_2d(Vector& u, int N, double coeff, Backend b = best_backend) {
  Vector lap(u.size());
  laplacian_stencil_2d_periodic(u, lap, N);
  axpy(coeff, lap, u, b);
}

inline void diffusion_step_2d_dirichlet(Vector& u,
                                        int N,
                                        double coeff,
                                        Backend b = best_backend) {
  Vector lap(u.size());
  laplacian_stencil_2d(u, lap, N);
  axpy(coeff, lap, u, b);
}

inline void diffusion_step_2d_4th_dirichlet(Vector& u,
                                            int N,
                                            double coeff,
                                            Backend b = best_backend) {
  Vector lap(u.size());
  laplacian_stencil_2d_4th(u, lap, N);
  axpy(coeff, lap, u, b);
}

inline void diffusion_step_2d_dirichlet(ScalarField2D& g,
                                        double coeff,
                                        Backend b = best_backend) {
  diffusion_step_2d_dirichlet(g.vec(), g.N(), coeff, b);
}

inline void diffusion_step_2d_4th_dirichlet(ScalarField2D& g,
                                            double coeff,
                                            Backend b = best_backend) {
  diffusion_step_2d_4th_dirichlet(g.vec(), g.N(), coeff, b);
}

inline SparseMatrix laplacian_sparse_2d(int N) {
  const int n = N * N;
  std::vector<idx> rows, cols;
  std::vector<real> vals;
  rows.reserve(5 * n);
  cols.reserve(5 * n);
  vals.reserve(5 * n);
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      int k = (i * N) + j;
      rows.push_back(k);
      cols.push_back(k);
      vals.push_back(-4.0);
      if (i > 0) {
        rows.push_back(k);
        cols.push_back(((i - 1) * N) + j);
        vals.push_back(1.0);
      }
      if (i < N - 1) {
        rows.push_back(k);
        cols.push_back(((i + 1) * N) + j);
        vals.push_back(1.0);
      }
      if (j > 0) {
        rows.push_back(k);
        cols.push_back((i * N) + (j - 1));
        vals.push_back(1.0);
      }
      if (j < N - 1) {
        rows.push_back(k);
        cols.push_back((i * N) + (j + 1));
        vals.push_back(1.0);
      }
    }
  }
  return SparseMatrix::from_triplets(n, n, rows, cols, vals);
}

inline SparseMatrix backward_euler_matrix(int N, double coeff) {
  const int n = N * N;
  std::vector<idx> rows, cols;
  std::vector<real> vals;
  rows.reserve(5 * n);
  cols.reserve(5 * n);
  vals.reserve(5 * n);
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      int k = (i * N) + j;
      rows.push_back(k);
      cols.push_back(k);
      vals.push_back(1.0 + (4.0 * coeff));
      if (i > 0) {
        rows.push_back(k);
        cols.push_back(((i - 1) * N) + j);
        vals.push_back(-coeff);
      }
      if (i < N - 1) {
        rows.push_back(k);
        cols.push_back(((i + 1) * N) + j);
        vals.push_back(-coeff);
      }
      if (j > 0) {
        rows.push_back(k);
        cols.push_back((i * N) + (j - 1));
        vals.push_back(-coeff);
      }
      if (j < N - 1) {
        rows.push_back(k);
        cols.push_back((i * N) + (j + 1));
        vals.push_back(-coeff);
      }
    }
  }
  return SparseMatrix::from_triplets(n, n, rows, cols, vals);
}

inline SparseMatrix backward_euler_matrix(const Grid2D& grid, double coeff) {
  return backward_euler_matrix(grid.N, coeff);
}

inline BackwardEulerOperator2D backward_euler_operator(int N, double coeff) {
  return BackwardEulerOperator2D(N, coeff);
}

inline BackwardEulerOperator2D backward_euler_operator(const Grid2D& grid, double coeff) {
  return backward_euler_operator(grid.N, coeff);
}

inline LinearSolver make_cg_solver(const SparseMatrix& A, real tol = 1e-6) {
  return [&A, tol](const Vector& rhs, Vector& x) {
    operators::SparseOp op(A);
    return cg(operators::assume_spd(op), rhs, x, tol);
  };
}

} // namespace num::pde

namespace num {
using pde::make_cg_solver;
}
