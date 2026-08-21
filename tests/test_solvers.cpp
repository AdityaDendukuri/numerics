#include "core/concepts.hpp"
#include "analysis/talbot.hpp"
#include "linalg/factorization/thomas.hpp"
#include "linalg/solvers/solvers.hpp"
#include "linalg/solvers/sparse_resolvent.hpp"
#include "linalg/solvers/dense_resolvent.hpp"
#include "linalg/sparse/sparse.hpp"
#include "linalg/sparse/sparse_op.hpp"
#include "operator/operator.hpp"

#include "pde/diffusion.hpp"
#include "solve/solve.hpp"
#include <cmath>
#include <gtest/gtest.h>

using namespace num;

template<class Op>
concept CgCallable = requires(const Op& A, const Vector& b, Vector& x) { cg(A, b, x); };

static_assert(!CgCallable<operators::DenseOp>);

// Conjugate Gradient

TEST(Resolvent, DenseSolve) {
  Matrix A(2, 2, 0.0);
  A(0, 0) = 1.0; A(0, 1) = 2.0;
  A(1, 0) = 3.0; A(1, 1) = 4.0;

  Vector b{1.0, 2.0};
  cplx s(2.0, 1.0);

  auto x = resolvent_solve(s, A, b);
  ASSERT_EQ(x.size(), 2);

  cplx res0 = (s - cplx(A(0,0), 0)) * x[0] - cplx(A(0,1), 0) * x[1];
  cplx res1 = -cplx(A(1,0), 0) * x[0] + (s - cplx(A(1,1), 0)) * x[1];

  EXPECT_NEAR(res0.real(), b[0], 1e-10);
  EXPECT_NEAR(res0.imag(), 0.0, 1e-10);
  EXPECT_NEAR(res1.real(), b[1], 1e-10);
  EXPECT_NEAR(res1.imag(), 0.0, 1e-10);
}


TEST(CG, Small3x3) {

  // A = [4 1 0; 1 4 1; 0 1 4], b = [1; 2; 3]  =>  x = [5/28, 2/7, 19/28]
  Matrix A(3, 3, 0.0);
  A(0, 0) = 4;
  A(0, 1) = 1;
  A(1, 0) = 1;
  A(1, 1) = 4;
  A(1, 2) = 1;
  A(2, 1) = 1;
  A(2, 2) = 4;

  Vector b{1.0, 2.0, 3.0};
  Vector x(3, 0.0);
  SolverResult r = cg(A, b, x);

  EXPECT_TRUE(r.converged);
  EXPECT_LT(r.residual, 1e-10);
  EXPECT_NEAR(x[0], 5.0 / 28.0, 1e-6);
  EXPECT_NEAR(x[1], 2.0 / 7.0, 1e-6);
  EXPECT_NEAR(x[2], 19.0 / 28.0, 1e-6);
}

TEST(CG, DiagonalDominant5x5) {
  idx n = 5;
  Matrix A(n, n, 0.0);
  for (idx i = 0; i < n; ++i) {
    A(i, i) = 10.0;
    if (i > 0) {
      A(i, i - 1) = 1.0;
}
    if (i < n - 1) {
      A(i, i + 1) = 1.0;
}
  }
  Vector b(n, 1.0), x(n, 0.0);
  SolverResult r = cg(A, b, x);

  EXPECT_TRUE(r.converged);
  EXPECT_LT(r.residual, 1e-10);

  Vector Ax(n);
  matvec(A, x, Ax);
  real err = 0;
  for (idx i = 0; i < n; ++i) {
    err += (Ax[i] - b[i]) * (Ax[i] - b[i]);
}
  EXPECT_LT(std::sqrt(err), 1e-9);
}

TEST(CG, ConvergesWithinN) {
  idx n = 10;
  Matrix A(n, n, 0.0);
  for (idx i = 0; i < n; ++i) {
    A(i, i) = static_cast<real>(i + 1);
}

  Vector b(n), x(n, 0.0);
  for (idx i = 0; i < n; ++i) {
    b[i] = static_cast<real>(i + 1);
}

  SolverResult r = cg(A, b, x);
  EXPECT_TRUE(r.converged);
  EXPECT_LE(r.iterations, n);
  for (idx i = 0; i < n; ++i) {
    EXPECT_NEAR(x[i], 1.0, 1e-9);
}
}

TEST(MatrixProperties, CheckedSymmetricAndSPD) {
  Matrix A(2, 2, 0.0);
  A(0, 0) = 2.0;
  A(0, 1) = -1.0;
  A(1, 0) = -1.0;
  A(1, 1) = 2.0;

  EXPECT_TRUE(linalg::is_symmetric(A));
  EXPECT_TRUE(linalg::is_spd(A));

  auto S = linalg::make_symmetric(A);
  auto P = linalg::make_spd(A);
  EXPECT_EQ(S.rows(), 2);
  EXPECT_EQ(P.cols(), 2);
}

TEST(MatrixProperties, CheckedConstructorsRejectInvalidInput) {
  Matrix nonsym(2, 2, 0.0);
  nonsym(0, 0) = 1.0;
  nonsym(0, 1) = 2.0;
  nonsym(1, 0) = 0.0;
  nonsym(1, 1) = 1.0;

  Matrix indefinite(2, 2, 0.0);
  indefinite(0, 0) = 1.0;
  indefinite(1, 1) = -1.0;

  EXPECT_FALSE(linalg::is_symmetric(nonsym));
  EXPECT_FALSE(linalg::is_spd(indefinite));
  EXPECT_THROW((void)linalg::make_symmetric(nonsym), std::invalid_argument);
  EXPECT_THROW((void)linalg::make_spd(indefinite), std::invalid_argument);
}

static_assert(VectorLike<Vector>);
static_assert(MutableVectorLike<Vector>);
static_assert(ContiguousVectorLike<Vector>);
static_assert(DenseMatrixLike<Matrix>);
static_assert(MutableDenseMatrixLike<Matrix>);
static_assert(ContiguousDenseMatrixLike<Matrix>);

TEST(CG, DenseOperator) {
  Matrix A(3, 3, 0.0);
  A(0, 0) = 4;
  A(0, 1) = 1;
  A(1, 0) = 1;
  A(1, 1) = 4;
  A(1, 2) = 1;
  A(2, 1) = 1;
  A(2, 2) = 4;

  operators::DenseOp op(A);
  static_assert(LinearOperator<operators::DenseOp>);
  static_assert(
    SymmetricLinearOperator<decltype(operators::assume_symmetric(op))>);
  static_assert(SPDLinearOperator<decltype(operators::assume_spd(op))>);

  Vector b{1.0, 2.0, 3.0};
  Vector x(3, 0.0);
  SolverResult r = cg(operators::assume_spd(op), b, x);

  EXPECT_TRUE(r.converged);
  EXPECT_LT(r.residual, 1e-10);
  EXPECT_NEAR(x[0], 5.0 / 28.0, 1e-6);
  EXPECT_NEAR(x[1], 2.0 / 7.0, 1e-6);
  EXPECT_NEAR(x[2], 19.0 / 28.0, 1e-6);
}

TEST(SolveDispatch, MatrixCGWithCheckedSPD) {
  Matrix A(3, 3, 0.0);
  A(0, 0) = 4;
  A(0, 1) = 1;
  A(1, 0) = 1;
  A(1, 1) = 4;
  A(1, 2) = 1;
  A(2, 1) = 1;
  A(2, 2) = 4;

  Vector b{1.0, 2.0, 3.0};
  const LinearSolution r = solve(LinearProblem{linalg::make_spd(A), b}, CG{});

  EXPECT_TRUE(r.converged);
  EXPECT_NEAR(r.u[0], 5.0 / 28.0, 1e-6);
  EXPECT_NEAR(r.u[1], 2.0 / 7.0, 1e-6);
  EXPECT_NEAR(r.u[2], 19.0 / 28.0, 1e-6);
}

TEST(SolveDispatch, DenseGMRES) {
  Matrix A(2, 2, 0.0);
  A(0, 0) = 3.0;
  A(0, 1) = 1.0;
  A(1, 0) = 0.0;
  A(1, 1) = 2.0;

  Vector b{5.0, 4.0};
  const LinearSolution r = solve(LinearProblem{A, b}, GMRES{.tol = 1e-12, .max_iter = 20});

  EXPECT_TRUE(r.converged);
  EXPECT_NEAR(r.u[0], 1.0, 1e-8);
  EXPECT_NEAR(r.u[1], 2.0, 1e-8);
}

TEST(CG, SparseOperator) {
  auto A = SparseMatrix::from_triplets(3,
                                       3,
                                       {0, 0, 1, 1, 1, 2, 2},
                                       {0, 1, 0, 1, 2, 1, 2},
                                       {4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0});

  operators::SparseOp op(A);
  static_assert(LinearOperator<operators::SparseOp>);
  static_assert(SPDLinearOperator<decltype(operators::assume_spd(op))>);

  Vector b{1.0, 2.0, 3.0};
  Vector x(3, 0.0);
  SolverResult r = cg(operators::assume_spd(op), b, x, 1e-10, 100, Backend::seq);

  EXPECT_TRUE(r.converged);
  EXPECT_LT(r.residual, 1e-10);
  EXPECT_NEAR(x[0], 5.0 / 28.0, 1e-6);
  EXPECT_NEAR(x[1], 2.0 / 7.0, 1e-6);
  EXPECT_NEAR(x[2], 19.0 / 28.0, 1e-6);
}

TEST(Operators, CallableOperator) {
  auto op = operators::make_op(
    [](const Vector& x, Vector& y) {
      y[0] = 2.0 * x[0];
      y[1] = 3.0 * x[1];
    },
    2);
  static_assert(LinearOperator<decltype(op)>);

  Vector x{4.0, 5.0};
  Vector y;
  op.apply(x, y);

  EXPECT_EQ(op.rows(), 2);
  EXPECT_EQ(op.cols(), 2);
  EXPECT_EQ(y.size(), 2);
  EXPECT_DOUBLE_EQ(y[0], 8.0);
  EXPECT_DOUBLE_EQ(y[1], 15.0);
}

TEST(PCG, JacobiPreconditioner) {
  auto A =
    SparseMatrix::from_triplets(4,
                                4,
                                {0, 0, 1, 1, 1, 2, 2, 2, 3, 3},
                                {0, 1, 0, 1, 2, 1, 2, 3, 2, 3},
                                {4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0});
  operators::SparseOp op(A);
  auto M = jacobi_preconditioner(A);
  Vector b{1.0, 2.0, 3.0, 4.0};
  Vector x(4, 0.0);

  SolverResult r = pcg(operators::assume_spd(op), M, b, x, 1e-10, 100);
  EXPECT_TRUE(r.converged);

  Vector Ax(4);
  sparse_matvec(A, x, Ax);
  for (idx i = 0; i < 4; ++i) {
    EXPECT_NEAR(Ax[i], b[i], 1e-9);
  }
}

TEST(MINRES, SymmetricIndefiniteOperator) {
  Matrix A(3, 3, 0.0);
  A(0, 0) = 2.0;
  A(1, 1) = -1.0;
  A(2, 2) = 3.0;

  operators::DenseOp op(A);
  Vector b{2.0, -2.0, 6.0};
  Vector x(3, 0.0);

  SolverResult r = minres(operators::assume_symmetric(op), b, x, 1e-10, 10);
  EXPECT_TRUE(r.converged);
  EXPECT_NEAR(x[0], 1.0, 1e-8);
  EXPECT_NEAR(x[1], 2.0, 1e-8);
  EXPECT_NEAR(x[2], 2.0, 1e-8);
}

TEST(PDEOperators, BackwardEulerOperatorIsSPD) {
  auto A = pde::backward_euler_operator(4, 0.1);
  static_assert(SPDLinearOperator<decltype(A)>);

  Vector b(A.rows(), 1.0);
  Vector x(A.rows(), 0.0);
  SolverResult r = cg(A, b, x, 1e-10, 100);

  EXPECT_TRUE(r.converged);
  EXPECT_LT(r.residual, 1e-10);
}

// Thomas algorithm

TEST(Thomas, Small4x4) {
  Vector a{-1.0, -1.0, -1.0}, b{2.0, 2.0, 2.0, 2.0}, c{-1.0, -1.0, -1.0};
  Vector d{1.0, 0.0, 0.0, 1.0}, x(4);
  thomas(a, b, c, d, x);
  for (idx i = 0; i < 4; ++i) {
    EXPECT_NEAR(x[i], 1.0, 1e-10);
}
}

TEST(Thomas, Laplacian1D) {
  idx n = 10;
  Vector a(n - 1, -1.0), b(n, 2.0), c(n - 1, -1.0), d(n, 1.0), x(n);
  thomas(a, b, c, d, x);
  for (idx i = 0; i < n; ++i) {
    real Ax = b[i] * x[i];
    if (i > 0) {
      Ax += a[i - 1] * x[i - 1];
}
    if (i < n - 1) {
      Ax += c[i] * x[i + 1];
}
    EXPECT_NEAR(Ax, d[i], 1e-10);
  }
}

TEST(Thomas, TwoByTwo) {
  Vector a{2.0}, b{3.0, 4.0}, c{1.0}, d{5.0, 6.0}, x(2);
  thomas(a, b, c, d, x);
  EXPECT_NEAR(x[0], 1.4, 1e-10);
  EXPECT_NEAR(x[1], 0.8, 1e-10);
}

// Gauss-Seidel

TEST(GaussSeidel, DiagonalDominant3x3) {
  // [4 1 0; 1 4 1; 0 1 4] x = [1; 2; 3]  =>  same solution as CG test
  Matrix A(3, 3, 0.0);
  A(0, 0) = 4;
  A(0, 1) = 1;
  A(1, 0) = 1;
  A(1, 1) = 4;
  A(1, 2) = 1;
  A(2, 1) = 1;
  A(2, 2) = 4;

  Vector b{1.0, 2.0, 3.0};
  Vector x(3, 0.0);
  SolverResult r = gauss_seidel(A, b, x);

  EXPECT_TRUE(r.converged);
  EXPECT_LT(r.residual, 1e-10);
  EXPECT_NEAR(x[0], 5.0 / 28.0, 1e-6);
  EXPECT_NEAR(x[1], 2.0 / 7.0, 1e-6);
  EXPECT_NEAR(x[2], 19.0 / 28.0, 1e-6);
}

TEST(GaussSeidel, DiagonalSystem) {
  // Diagonal A: solution is trivially b[i]/A[i][i]
  idx n = 8;
  Matrix A(n, n, 0.0);
  for (idx i = 0; i < n; ++i) {
    A(i, i) = static_cast<real>(i + 1);
}

  Vector b(n), x(n, 0.0);
  for (idx i = 0; i < n; ++i) {
    b[i] = static_cast<real>((i + 1) * (i + 1));
}

  SolverResult r = gauss_seidel(A, b, x);
  EXPECT_TRUE(r.converged);
  for (idx i = 0; i < n; ++i) {
    EXPECT_NEAR(x[i], static_cast<real>(i + 1), 1e-8);
}
}

TEST(GaussSeidel, ResidualVerified) {
  idx n = 6;
  Matrix A(n, n, 0.0);
  for (idx i = 0; i < n; ++i) {
    A(i, i) = 8.0;
    if (i > 0) {
      A(i, i - 1) = -1.0;
}
    if (i < n - 1) {
      A(i, i + 1) = -1.0;
}
  }
  Vector b(n, 1.0), x(n, 0.0);
  SolverResult r = gauss_seidel(A, b, x);

  EXPECT_TRUE(r.converged);
  EXPECT_LT(r.residual, 1e-10);

  // Verify Ax ~= b
  Vector Ax(n);
  matvec(A, x, Ax);
  for (idx i = 0; i < n; ++i) {
    EXPECT_NEAR(Ax[i], b[i], 1e-8);
}
}

// Jacobi

TEST(Jacobi, DiagonalDominant3x3) {
  Matrix A(3, 3, 0.0);
  A(0, 0) = 4;
  A(0, 1) = 1;
  A(1, 0) = 1;
  A(1, 1) = 4;
  A(1, 2) = 1;
  A(2, 1) = 1;
  A(2, 2) = 4;

  Vector b{1.0, 2.0, 3.0};
  Vector x(3, 0.0);
  SolverResult r = jacobi(A, b, x);

  EXPECT_TRUE(r.converged);
  EXPECT_LT(r.residual, 1e-10);
  EXPECT_NEAR(x[0], 5.0 / 28.0, 1e-6);
  EXPECT_NEAR(x[1], 2.0 / 7.0, 1e-6);
  EXPECT_NEAR(x[2], 19.0 / 28.0, 1e-6);
}

TEST(Jacobi, DiagonalSystem) {
  idx n = 8;
  Matrix A(n, n, 0.0);
  for (idx i = 0; i < n; ++i) {
    A(i, i) = static_cast<real>(i + 1);
}

  Vector b(n), x(n, 0.0);
  for (idx i = 0; i < n; ++i) {
    b[i] = static_cast<real>((i + 1) * (i + 1));
}

  // Diagonal system: Jacobi converges in one iteration
  SolverResult r = jacobi(A, b, x, 1e-10, 1);
  EXPECT_EQ(r.iterations, static_cast<idx>(1));
  for (idx i = 0; i < n; ++i) {
    EXPECT_NEAR(x[i], static_cast<real>(i + 1), 1e-10);
}
}

TEST(Jacobi, ResidualVerified) {
  idx n = 6;
  Matrix A(n, n, 0.0);
  for (idx i = 0; i < n; ++i) {
    A(i, i) = 8.0;
    if (i > 0) {
      A(i, i - 1) = -1.0;
}
    if (i < n - 1) {
      A(i, i + 1) = -1.0;
}
  }
  Vector b(n, 1.0), x(n, 0.0);
  SolverResult r = jacobi(A, b, x);

  EXPECT_TRUE(r.converged);
  EXPECT_LT(r.residual, 1e-10);

  Vector Ax(n);
  matvec(A, x, Ax);
  for (idx i = 0; i < n; ++i) {
    EXPECT_NEAR(Ax[i], b[i], 1e-8);
}
}

// GMRES (Krylov)

TEST(GMRES, SPD3x3Dense) {
  // Same SPD system  -- GMRES should also solve it
  Matrix A(3, 3, 0.0);
  A(0, 0) = 4;
  A(0, 1) = 1;
  A(1, 0) = 1;
  A(1, 1) = 4;
  A(1, 2) = 1;
  A(2, 1) = 1;
  A(2, 2) = 4;

  Vector b{1.0, 2.0, 3.0};
  Vector x(3, 0.0);
  SolverResult r = gmres(A, b, x);

  EXPECT_TRUE(r.converged);
  EXPECT_LT(r.residual, 1e-6);
  EXPECT_NEAR(x[0], 5.0 / 28.0, 1e-5);
  EXPECT_NEAR(x[1], 2.0 / 7.0, 1e-5);
  EXPECT_NEAR(x[2], 19.0 / 28.0, 1e-5);
}

TEST(GMRES, DenseOperator) {
  Matrix A(3, 3, 0.0);
  A(0, 0) = 4;
  A(0, 1) = 1;
  A(1, 0) = 1;
  A(1, 1) = 4;
  A(1, 2) = 1;
  A(2, 1) = 1;
  A(2, 2) = 4;

  operators::DenseOp op(A);

  Vector b{1.0, 2.0, 3.0};
  Vector x(3, 0.0);
  SolverResult r = gmres(op, b, x);

  EXPECT_TRUE(r.converged);
  EXPECT_LT(r.residual, 1e-6);
  EXPECT_NEAR(x[0], 5.0 / 28.0, 1e-5);
  EXPECT_NEAR(x[1], 2.0 / 7.0, 1e-5);
  EXPECT_NEAR(x[2], 19.0 / 28.0, 1e-5);
}

TEST(GMRES, NonSymmetricDense) {
  // Non-symmetric system: A = [3 1; 1 2], b = [5; 3]  =>  x = [1, 2]
  Matrix A(2, 2, 0.0);
  A(0, 0) = 3;
  A(0, 1) = 1;
  A(1, 0) = 1;
  A(1, 1) = 2;

  Vector b{5.0, 3.0}; // actually symmetric here but checks general path
  Vector x(2, 0.0);
  SolverResult r = gmres(A, b, x);

  EXPECT_TRUE(r.converged);
  EXPECT_NEAR(x[0], 1.4, 1e-5);
  EXPECT_NEAR(x[1], 0.8, 1e-5);
}

TEST(GMRES, SparseLaplacian1D) {
  // 1D Laplacian on 10 nodes via SparseMatrix
  idx n = 10;
  std::vector<idx> rows, cols;
  std::vector<real> vals;
  for (idx i = 0; i < n; ++i) {
    rows.push_back(i);
    cols.push_back(i);
    vals.push_back(2.0);
    if (i > 0) {
      rows.push_back(i);
      cols.push_back(i - 1);
      vals.push_back(-1.0);
    }
    if (i < n - 1) {
      rows.push_back(i);
      cols.push_back(i + 1);
      vals.push_back(-1.0);
    }
  }
  SparseMatrix A = SparseMatrix::from_triplets(n, n, rows, cols, vals);

  Vector b(n, 1.0), x(n, 0.0);
  SolverResult r = gmres(A, b, x);

  EXPECT_TRUE(r.converged);
  EXPECT_LT(r.residual, 1e-6);

  // Verify Ax ~= b
  Vector Ax(n);
  sparse_matvec(A, x, Ax);
  for (idx i = 0; i < n; ++i) {
    EXPECT_NEAR(Ax[i], b[i], 1e-5);
}
}

TEST(GMRES, MatrixFree) {
  idx n = 5;
  Vector diag(n);
  for (idx i = 0; i < n; ++i) {
    diag[i] = static_cast<real>(i + 1);
}

  auto op = operators::make_op(
    [&](const Vector& in, Vector& out) {
      out = Vector(n);
      for (idx i = 0; i < n; ++i) {
        out[i] = diag[i] * in[i];
}
    },
    n);

  Vector b(n, 1.0), x(n, 0.0);
  SolverResult r = gmres(op, b, x);

  EXPECT_TRUE(r.converged);
  for (idx i = 0; i < n; ++i) {
    EXPECT_NEAR(x[i], 1.0 / static_cast<real>(i + 1), 1e-5);
}
}

// SparseMatrix construction

TEST(SparseMatrix, FromTriplets) {
  // 3x3 identity
  SparseMatrix I =
    SparseMatrix::from_triplets(3, 3, {0, 1, 2}, {0, 1, 2}, {1.0, 1.0, 1.0});
  EXPECT_EQ(I.nnz(), static_cast<idx>(3));
  EXPECT_NEAR(I(0, 0), 1.0, 1e-15);
  EXPECT_NEAR(I(1, 1), 1.0, 1e-15);
  EXPECT_NEAR(I(0, 1), 0.0, 1e-15);
}

TEST(SparseMatrix, FromCSC) {
  // CSC for [1 0 2; 0 3 0], with an Armadillo-style trailing payload item.
  const SparseMatrix A = SparseMatrix::from_csc(
    2, 3,
    {1.0, 3.0, 2.0, 99.0},
    {0, 1, 0, 999},
    {0, 1, 2, 3});
  EXPECT_EQ(A.nnz(), static_cast<idx>(3));
  EXPECT_NEAR(A(0, 0), 1.0, 1e-15);
  EXPECT_NEAR(A(1, 1), 3.0, 1e-15);
  EXPECT_NEAR(A(0, 2), 2.0, 1e-15);
  EXPECT_NEAR(A(1, 0), 0.0, 1e-15);
}

TEST(SparseMatrix, DuplicatesSummed) {
  // Two entries at (0,0): should be summed to 3.0
  SparseMatrix A =
    SparseMatrix::from_triplets(2, 2, {0, 0, 1}, {0, 0, 1}, {1.0, 2.0, 4.0});
  EXPECT_NEAR(A(0, 0), 3.0, 1e-15);
  EXPECT_NEAR(A(1, 1), 4.0, 1e-15);
}

TEST(SparseMatrix, Matvec) {
  // A = [2 -1; -1 2], x = [1; 1]  =>  y = [1; 1]
  SparseMatrix A =
    SparseMatrix::from_triplets(2, 2, {0, 0, 1, 1}, {0, 1, 0, 1}, {2.0, -1.0, -1.0, 2.0});
  Vector x{1.0, 1.0}, y(2);
  sparse_matvec(A, x, y);
  EXPECT_NEAR(y[0], 1.0, 1e-14);
  EXPECT_NEAR(y[1], 1.0, 1e-14);
}

TEST(Resolvent, ReusableFactorAndBatch) {
  Matrix A(2, 2);
  A(0, 0) = 1.0; A(0, 1) = 0.0;
  A(1, 0) = 0.0; A(1, 1) = 2.0;
  ResolventFactor factor(cplx(3.0, 0.0), A);
  const auto x = factor.solve(std::vector<cplx>{cplx(2.0), cplx(6.0)});
  EXPECT_NEAR(x[0].real(), 1.0, 1e-12);
  EXPECT_NEAR(x[1].real(), 6.0, 1e-12);
}

TEST(Talbot, NodesScaleWithTime) {
  const auto a = talbot_nodes(1.0, 8);
  const auto b = talbot_nodes(2.0, 8);
  ASSERT_EQ(a.size(), b.size());
  for (idx k = 0; k < a.size(); ++k) {
    EXPECT_NEAR((a[k].shift / b[k].shift).real(), 2.0, 1e-12);
    EXPECT_NEAR((a[k].weight / b[k].weight).real(), 2.0, 1e-12);
  }
}

TEST(SparseResolvent, OptionalBackend) {
  SparseMatrix A = SparseMatrix::from_triplets(2, 2, {0, 1}, {0, 1}, {2.0, 3.0});
  SparseResolventSolver solver(A);
  if (!sparse_resolvent_available()) {
    EXPECT_THROW(solver.factorize(cplx(1.0)), std::runtime_error);
    return;
  }
  solver.factorize(cplx(1.0));
  const auto x1 = solver.solve(std::vector<cplx>{cplx(1.0), cplx(2.0)});
  EXPECT_NEAR(x1[0].real(), -1.0, 1e-12);
  EXPECT_NEAR(x1[1].real(), -1.0, 1e-12);
  solver.factorize(cplx(4.0));
  const auto x2 = solver.solve(std::vector<cplx>{cplx(2.0), cplx(1.0)});
  EXPECT_NEAR(x2[0].real(), 1.0, 1e-12);
  EXPECT_NEAR(x2[1].real(), 1.0, 1e-12);

  SparseResolventSolver symmetric_solver(A, {.symmetric_pattern = true});
  symmetric_solver.factorize(cplx(4.0));
  const auto x3 = symmetric_solver.solve(std::vector<cplx>{cplx(2.0), cplx(1.0)});
  EXPECT_NEAR(x3[0].real(), 1.0, 1e-12);
  EXPECT_NEAR(x3[1].real(), 1.0, 1e-12);
}

TEST(DenseResolvent, ReusableFactorization) {
  const SparseMatrix matrix =
      SparseMatrix::from_triplets(2, 2, {0, 0, 1}, {0, 1, 1}, {2.0, 1.0, 3.0});
  DenseResolventSolver solver(matrix);

  solver.factorize(cplx(4.0, 1.0));
  const std::vector<cplx> expected{cplx(1.0, -0.5), cplx(-0.25, 0.75)};
  std::vector<cplx> rhs(2);
  rhs[0] = (cplx(4.0, 1.0) - 2.0) * expected[0] - expected[1];
  rhs[1] = (cplx(4.0, 1.0) - 3.0) * expected[1];
  const auto solution = solver.solve(rhs);

  EXPECT_NEAR(solution[0].real(), expected[0].real(), 1e-12);
  EXPECT_NEAR(solution[0].imag(), expected[0].imag(), 1e-12);
  EXPECT_NEAR(solution[1].real(), expected[1].real(), 1e-12);
  EXPECT_NEAR(solution[1].imag(), expected[1].imag(), 1e-12);
}
