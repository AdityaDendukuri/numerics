# Linear Algebra {#page_linalg}

## LU Factorization

### Factor a general dense matrix

```cpp
num::Matrix A = make_matrix();
num::LUResult factor = num::lu(A); // Store P*A=L*U.

if (factor.singular) {
    // A cannot be solved with this factor.
}
```

### Solve a system

```cpp
num::Vector b{5.0, 6.0, 5.0};
num::Vector x;

num::lu_solve(factor, b, x); // Solve A*x=b.
```

### Solve a transposed system

```cpp
num::Vector x;
num::lu_solve_transpose(factor, b, x); // Solve A^T*x=b.
```

### Solve several right-hand sides

```cpp
num::Matrix B = num::identity_columns(A.rows(), 0, 2);
num::Matrix X;

num::lu_solve(factor, B, X); // Solve A*X=B with one factor.
```

### Overwrite a right-hand side

```cpp
num::Vector x = b;
num::solve_in_place(factor, x); // Replace x with A^-1*b.
```

### Determinant and inverse

```cpp
num::real determinant = num::lu_det(factor);
num::Matrix inverse = num::lu_inv(factor); // Solve A*X=I.
```

## Cholesky Factorization

### Validate and factor an SPD matrix

```cpp
auto spd = num::linalg::make_spd(A); // Check A=A^T>0.
num::CholeskyResult factor = num::cholesky(spd); // Store A=L*L^T.
```

Declare an invariant that construction already guarantees:

```cpp
auto spd = num::linalg::assume_spd(A); // No numerical validation.
auto factor = num::cholesky(spd);
```

### Solve with Cholesky

```cpp
num::Vector x;
num::cholesky_solve(factor, b, x); // Solve A*x=b.
```

### Solve several SPD systems

```cpp
num::Matrix X;
num::cholesky_solve(factor, B, X); // Solve A*X=B.
```

### Apply a rank-one update

```cpp
num::Vector u{0.1, 0.0, -0.1};
num::cholesky_update(factor, u); // Factor A+u*u^T in O(n^2).
```

### Apply a rank-one downdate

```cpp
num::cholesky_downdate(factor, u); // Factor A-u*u^T.
```

`cholesky_downdate` throws when the downdated matrix is not positive definite.

## QR and Least Squares

### Factor a rectangular matrix

```cpp
num::Matrix A(4, 2, 0.0);
// Fill four observations of two predictors.

num::QRResult factor = num::qr(A); // A=Q*R.
```

### Solve a least-squares problem

```cpp
num::Vector observations{1.0, 2.0, 2.0, 4.0};
num::Vector coefficients;

num::qr_solve(factor, observations, coefficients);
// coefficients minimizes ||A*x-observations||_2.
```

## Tridiagonal Systems

### Solve a real tridiagonal system

```cpp
num::Vector lower{-1.0, -1.0};
num::Vector diagonal{2.0, 2.0, 2.0};
num::Vector upper{-1.0, -1.0};
num::Vector rhs{1.0, 0.0, 1.0};
num::Vector x;

num::thomas(lower, diagonal, upper, rhs, x); // O(n) solve.
```

### Reuse a complex tridiagonal factor

```cpp
num::ComplexTriDiag factor;
factor.factor(64, {-1.0, 0.0}, {2.0, 0.1}, {-1.0, 0.0});
```

```cpp
std::vector<num::cplx> rhs(64, 1.0);
factor.solve(rhs); // Replace rhs with the solution.
```

## Banded Matrices

### Construct a banded matrix

```cpp
num::BandedMatrix A(128, 1, 1);
// 128 rows, one lower diagonal, one upper diagonal.
```

```cpp
for (num::idx i = 0; i < A.rows(); ++i) {
    A(i, i) = 2.0;
    if (i > 0) A(i, i - 1) = -1.0;
    if (i + 1 < A.rows()) A(i, i + 1) = -1.0;
}
```

### Solve once

```cpp
num::Vector b(A.rows(), 1.0);
num::Vector x;

num::BandedSolverResult result = num::banded_solve(A, b, x);
```

### Factor once and solve repeatedly

```cpp
num::BandedMatrix factor = A;
std::vector<num::idx> pivots(A.rows());
num::banded_lu(factor, pivots.data());
```

```cpp
num::Vector x = b;
num::banded_lu_solve(factor, pivots.data(), x); // Overwrite x.
```

### Multiply by a banded matrix

```cpp
num::Vector y(A.rows(), 0.0);
num::banded_matvec(A, x, y); // y <- A*x.
```

```cpp
num::banded_gemv(2.0, A, x, 0.5, y); // y <- 2*A*x+0.5*y.
```

### Estimate conditioning

```cpp
num::real norm1 = num::banded_norm1(A);
num::real rcond = num::banded_rcond(factor, pivots.data(), norm1);
```

## Sparse Direct Solvers

### Check backend availability

```cpp
bool has_klu = num::klu_available();
bool has_umfpack = num::umfpack_available();
```

### Factor and solve with KLU

```cpp
num::SparseMatrix A = make_sparse_matrix();
num::KLUFactor factor(A);

num::Vector b(A.n_rows(), 1.0);
num::Vector x;
factor.solve(b, x); // Solve A*x=b.
```

### Solve a sparse transposed system

```cpp
factor.solve_transpose(b, x); // Solve A^T*x=b.
```

### Select a dense or sparse factor automatically

```cpp
num::AutoLinearSolver factor(A, {.dense_limit = 32});
// Small systems use dense LU; larger systems use KLU when available.
```

```cpp
factor.solve(b, x);
factor.solve_transpose(b, x);
```

## Selected Inverse Entries

### Create reusable workspace

```cpp
auto factor = num::lu(A);
num::InverseDiagonalWorkspace workspace;
```

### Compute the inverse diagonal

```cpp
num::Vector diagonal(A.rows(), 0.0);
num::inverse_diagonal(factor, diagonal, workspace);
// Solves blocked columns of A*X=I without storing the full inverse.
```

### Request arbitrary inverse entries

```cpp
std::array<num::idx, 2> rows{0, 2};
std::array<num::idx, 2> columns{1, 2};
num::Vector entries(2, 0.0);

num::selected_inverse(factor, rows, columns, entries, workspace);
// entries == {A^-1(0,1), A^-1(2,2)}.
```

### Request a principal inverse block

```cpp
std::array<num::idx, 2> indices{0, 2};
num::Matrix block;

num::inverse_principal_block(factor, indices, block, workspace);
// block == A^-1(indices, indices).
```

### Update an inverse diagonal after a low-rank change

```cpp
num::Matrix U(A.rows(), 1, 0.0);
U(0, 0) = 1.0;
num::Matrix W(1, 1, 0.25);
// updated_A = A+U*W*U^T.
```

```cpp
auto updated_factor = num::lu(updated_A);
num::Vector updated_diagonal(A.rows(), 0.0);

auto path = num::inverse_diagonal_after_update(
    factor, updated_factor, diagonal, U, W,
    updated_diagonal, workspace);
```

`path` is `woodbury` when the reduced update is safe and `direct` when the
updated factor is used for blocked identity solves.

## Iterative Solvers

### Wrap a sparse matrix as an operator

```cpp
num::operators::SparseOp Aop(A_sparse);
num::Vector x(A_sparse.n_rows(), 0.0); // Initial guess.
```

### Conjugate gradients

```cpp
auto spd = num::operators::assume_spd(Aop);
num::SolverResult result = num::cg(spd, b, x, 1e-10, 1000);
```

### Preconditioned conjugate gradients

```cpp
auto M = num::jacobi_preconditioner(A_sparse);
num::SolverResult result = num::pcg(spd, M, b, x, 1e-10, 1000);
```

### MINRES for symmetric indefinite systems

```cpp
auto symmetric = num::operators::assume_symmetric(Aop);
num::SolverResult result = num::minres(symmetric, b, x, 1e-10, 1000);
```

### GMRES for general systems

```cpp
num::SolverResult result = num::gmres(Aop, b, x, 1e-8, 1000, 40);
// Restart after 40 Krylov vectors.
```

### Read convergence information

```cpp
bool converged = result.converged;
num::idx iterations = result.iterations;
num::real residual = result.residual;
```

### Use the problem-level solve interface

```cpp
auto problem = num::LinearProblem{spd, b};
num::LinearSolution solution = num::solve(
    problem, num::CG{.tol = 1e-10, .max_iter = 1000});
```

### Reuse a warm-started solve cache

```cpp
auto cache = num::init(problem, num::CG{}, solution.u);
num::LinearSolution repeated = num::solve(cache);
// cache.u remains the next initial guess.
```

## Eigenvalues

### Compute all symmetric eigenpairs

```cpp
num::EigenResult result = num::eig_sym(A);
num::Vector eigenvalues = result.values;
num::Matrix eigenvectors = result.vectors;
```

### Compute the dominant eigenpair

```cpp
num::PowerResult dominant = num::power_iteration(A);
num::real eigenvalue = dominant.eigenvalue;
num::Vector eigenvector = dominant.eigenvector;
```

### Find an eigenvalue near a shift

```cpp
num::PowerResult nearby = num::inverse_iteration(A, 2.0);
// Targets an eigenvalue near 2.
```

### Refine a supplied eigenvector

```cpp
num::PowerResult refined = num::rayleigh_iteration(A, dominant.eigenvector);
```

### Compute a few eigenpairs with Lanczos

```cpp
num::operators::DenseOp Aop(A);
auto symmetric = num::operators::assume_symmetric(Aop);

num::LanczosResult largest = num::lanczos(symmetric, 4);
// Return four Ritz pairs without a full decomposition.
```

## Singular Value Decomposition

### Full SVD

```cpp
num::SVDResult result = num::svd(A);
// A == result.U*diag(result.S)*result.Vt.
```

### Truncated randomized SVD

```cpp
num::SVDResult rank_eight = num::svd_truncated(A, 8);
```

### Reproducible truncated SVD

```cpp
num::Rng rng(1234);
auto result = num::svd_truncated(A, 8, num::blas, 10, &rng);
// Rank 8, 10 oversampling vectors, caller-owned RNG.
```

## Shifted Resolvents

### One dense shifted solve

```cpp
num::cplx shift{1.0, 2.0};
auto x = num::resolvent_solve(shift, A, b);
// x == (shift*I-A)^-1*b.
```

### Reuse one dense shifted factor

```cpp
num::ResolventFactor factor(shift, A);
std::vector<num::cplx> rhs(b.size(), 1.0);

auto x = factor.solve(rhs);
```

### Solve several shifts

```cpp
std::vector<num::cplx> shifts{{1.0, 0.0}, {1.0, 1.0}};
auto solutions = num::resolvent_solve_batch(shifts, A, b);
```

### Select a dense or sparse resolvent backend

```cpp
num::AutoResolventSolver factor(A_sparse, {.dense_limit = 128});
factor.factorize(shift);

std::vector<num::cplx> x;
factor.solve(rhs, x);
```

## Matrix Exponential Actions

### Dense operator

```cpp
num::operators::DenseOp Aop(A);
num::Vector y = num::expv(0.01, Aop, b, 30, 1e-8);
// y approximates exp(0.01*A)*b using a 30-vector Arnoldi basis.
```

### Sparse matrix

```cpp
num::Vector y = num::expv(0.01, A_sparse, b, 30, 1e-8);
// The matrix exponential is never materialized.
```

## Complete Programs

@example 01_direct_factorizations.cpp
@example 02_iterative_krylov_solvers.cpp
@example 03_resolvent_and_expv.cpp
@example 04_eigen_and_svd.cpp
@example 10_banded_and_spd_operators.cpp
