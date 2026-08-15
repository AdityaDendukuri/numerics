# Linear Algebra Examples {#page_linalg}

Direct solvers, Krylov iterations, eigenvalue routines, SVD, and matrix
functions use explicit numerical routine names. Operator arguments enter where
only the action \(y = Ax\) is required.

## Dense Direct Solve

```cpp
#include <numerics.hpp>

num::Matrix A(3, 3, 0.0);
A(0, 0) = 4.0; A(0, 1) = 1.0; A(0, 2) = 0.0;
A(1, 0) = 1.0; A(1, 1) = 3.0; A(1, 2) = 1.0;
A(2, 0) = 0.0; A(2, 1) = 1.0; A(2, 2) = 2.0;

num::Vector b{1.0, 2.0, 0.0};

auto F = num::lu(A);
num::Vector x;
num::lu_solve(F, b, x);
```

Use `lu_solve(F, B, X)` for multiple right-hand sides after one factorization.

For \f$A=A^T>0\f$:

```cpp
auto Aspd = num::linalg::make_spd(A);
auto C = num::cholesky(Aspd);
num::Vector x_spd(A.rows(), 0.0);
num::cholesky_solve(C, b, x_spd);
```

Use `num::linalg::assume_spd(A)` when the construction of \f$A\f$ already
guarantees symmetry and positive definiteness.

## Least Squares

```cpp
num::Matrix A(4, 2, 0.0);
A(0, 0) = 1.0; A(0, 1) = 0.0;
A(1, 0) = 1.0; A(1, 1) = 1.0;
A(2, 0) = 1.0; A(2, 1) = 2.0;
A(3, 0) = 1.0; A(3, 1) = 3.0;

num::Vector b{1.0, 2.0, 2.0, 4.0};
auto Q = num::qr(A);

num::Vector coeff;
num::qr_solve(Q, b, coeff);
```

This computes the minimizer of \(\|Ax-b\|_2\).

## Tridiagonal and Banded Systems

```cpp
const num::idx n = 128;
num::Vector a(n - 1, -1.0);
num::Vector d(n, 2.0);
num::Vector c(n - 1, -1.0);
num::Vector rhs(n, 1.0);
num::Vector x;

num::thomas(a, d, c, rhs, x);
```

For general banded storage:

```cpp
num::BandedMatrix A(n, 1, 1);
for (num::idx i = 0; i < n; ++i) {
    A(i, i) = 2.0;
    if (i > 0) A(i, i - 1) = -1.0;
    if (i + 1 < n) A(i, i + 1) = -1.0;
}

num::Vector xb;
auto info = num::banded_solve(A, rhs, xb);
```

## Sparse Krylov Solve

```cpp
std::vector<num::idx> rows, cols;
std::vector<num::real> vals;

auto push = [&](num::idx i, num::idx j, num::real v) {
    rows.push_back(i);
    cols.push_back(j);
    vals.push_back(v);
};

for (num::idx i = 0; i < n; ++i) {
    push(i, i, 2.0);
    if (i > 0) push(i, i - 1, -1.0);
    if (i + 1 < n) push(i, i + 1, -1.0);
}

num::SparseMatrix A = num::SparseMatrix::from_triplets(n, n, rows, cols, vals);
num::operators::SparseOp Aop(A);

num::Vector x(n, 0.0);
num::SolverResult r =
    num::cg(num::operators::assume_spd(Aop), rhs, x, 1e-10, 1000);
```

Use `gmres(Aop, b, x, tol, max_iter, restart)` for nonsymmetric systems.

The unified `solve(problem, algorithm)` form keeps the method choice explicit:

```cpp
auto Aspd = num::operators::assume_spd(Aop);
num::LinearSolution r =
    num::solve(num::LinearProblem{Aspd, rhs}, num::CG{.tol = 1e-10, .max_iter = 1000});

num::LinearSolution g =
    num::solve(num::LinearProblem{Aop, rhs}, num::GMRES{.tol = 1e-8, .restart = 40});
```

`r.u` is the solution vector; `r.iterations`, `r.residual`, and `r.converged`
report progress.

Use PCG when a preconditioner is available:

```cpp
auto M = num::jacobi_preconditioner(A);
num::SolverResult p =
    num::pcg(num::operators::assume_spd(Aop), M, rhs, x, 1e-10, 1000);
```

Use MINRES for symmetric indefinite operators:

```cpp
auto Sop = num::operators::assume_symmetric(Aop);
num::SolverResult m = num::minres(Sop, rhs, x, 1e-10, 1000);
```

## Eigenvalues and SVD

```cpp
auto E = num::eig_sym(A_dense);
auto dominant = num::power_iteration(A_dense);

auto Aop = num::operators::DenseOp(A_dense);
auto Ritz = num::lanczos(num::operators::assume_symmetric(Aop), 4);

auto S = num::svd(A_dense);
auto Sk = num::svd_truncated(A_dense, 8);
```

`eig_sym` returns a dense symmetric eigendecomposition. `lanczos` returns a
small set of Ritz pairs from a dense, sparse, or matrix-free symmetric
operator.

## Matrix Exponential Action

```cpp
auto Aop = num::operators::DenseOp(A_dense);
num::Vector y = num::expv(0.01, Aop, v, 30, 1e-8);
```

This forms \f$\exp(tA)v\f$ by Arnoldi projection without forming \f$\exp(A)\f$.
