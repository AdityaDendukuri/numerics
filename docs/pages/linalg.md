# Linear Algebra Examples {#page_linalg}

Numerics separates factorization from solving. Reuse a factor whenever the
matrix stays fixed, and select an operator-based solver when only products
with the matrix are available.

## LU Factorization

```cpp
num::Matrix A(3, 3, 0.0);
A(0, 0) = 4.0; A(0, 1) = 1.0;
A(1, 0) = 1.0; A(1, 1) = 4.0; A(1, 2) = 1.0;
A(2, 1) = 1.0; A(2, 2) = 4.0;
num::Vector b{5.0, 6.0, 5.0};

auto F = num::lu(A); // Store PA=LU once.

num::Vector x;
num::lu_solve(F, b, x); // Solve Ax=b.

num::Vector xt;
num::lu_solve_transpose(F, b, xt); // Solve A^T x=b.

double determinant = num::lu_det(F);
num::Matrix inverse = num::lu_inv(F); // Solve AX=I.
```

Several right-hand sides share the same factor:

```cpp
num::Matrix B = num::identity_columns(A.rows(), 0, 2);
num::Matrix X;
num::lu_solve(F, B, X);
num::lu_solve_transpose(F, B, X);

num::solve_in_place(F, b); // Replace b with A^-1 b.
```

Check `F.singular` before solving when singular input is possible.

## Cholesky Factorization and Rank-One Changes

```cpp
auto checked_A = num::linalg::make_spd(A); // Validate A=A^T>0.
auto F = num::cholesky(checked_A);         // Store A=LL^T.

num::Vector x;
num::cholesky_solve(F, b, x);

num::Matrix X;
num::cholesky_solve(F, B, X); // Several right-hand sides.
num::solve_in_place(F, b);
```

Update a factor in quadratic time when the matrix changes by one outer
product:

```cpp
num::Vector u{0.1, 0.0, -0.1};
num::cholesky_update(F, u);   // Factor A+u*u^T.
num::cholesky_downdate(F, u); // Return to A; throws if not SPD.
```

Use `linalg::assume_spd(A)` when construction already guarantees the property.

## QR and Least Squares

```cpp
num::Matrix A(4, 2, 0.0);
A(0, 0) = 1.0; A(0, 1) = 0.0;
A(1, 0) = 1.0; A(1, 1) = 1.0;
A(2, 0) = 1.0; A(2, 1) = 2.0;
A(3, 0) = 1.0; A(3, 1) = 3.0;

num::Vector observations{1.0, 2.0, 2.0, 4.0};
auto F = num::qr(A); // A=QR.

num::Vector coefficients;
num::qr_solve(F, observations, coefficients); // Minimize ||Ax-b||_2.
```

## Tridiagonal Systems

```cpp
num::Vector lower{-1.0, -1.0};
num::Vector diagonal{2.0, 2.0, 2.0};
num::Vector upper{-1.0, -1.0};
num::Vector rhs{1.0, 0.0, 1.0};
num::Vector x;

num::thomas(lower, diagonal, upper, rhs, x); // O(n) solve.
```

For repeated constant-coefficient complex systems:

```cpp
num::ComplexTriDiag F;
F.factor(64, {-1.0, 0.0}, {2.0, 0.1}, {-1.0, 0.0});

std::vector<num::cplx> rhs(64, 1.0);
F.solve(rhs); // Replace rhs with the solution.
```

## General Banded Systems

```cpp
num::BandedMatrix A(128, 1, 1); // n, lower bandwidth, upper bandwidth
for (num::idx i = 0; i < A.rows(); ++i) {
    A(i, i) = 2.0;
    if (i > 0) A(i, i - 1) = -1.0;
    if (i + 1 < A.rows()) A(i, i + 1) = -1.0;
}

num::Vector b(A.rows(), 1.0);
num::Vector x;
auto info = num::banded_solve(A, b, x);
```

Factor once for repeated solves:

```cpp
std::vector<num::idx> pivots(A.rows());
num::BandedMatrix LU = A;
auto info = num::banded_lu(LU, pivots.data());

num::Vector x = b;
num::banded_lu_solve(LU, pivots.data(), x);
```

Additional band helpers:

```cpp
num::banded_matvec(A, x, b);             // b <- A*x
num::banded_gemv(2.0, A, x, 0.5, b);    // b <- 2*A*x+0.5*b
double norm1 = num::banded_norm1(A);
double reciprocal_condition = num::banded_rcond(LU, pivots.data(), norm1);
```

## Sparse Direct Factors

```cpp
num::SparseMatrix A = make_sparse_matrix();
num::Vector b(A.n_rows(), 1.0);

if (num::klu_available()) {
    num::KLUFactor F(A);
    num::Vector x;
    F.solve(b, x);
    F.solve_transpose(b, x);
}
```

`KLUFactor` also accepts matrix right-hand sides. `UMFPACKFactor` provides the
same regular solve interface when `umfpack_available()` is true.

Select dense LU for small CSR matrices and KLU for larger ones:

```cpp
num::AutoLinearSolver F(A, {.dense_limit = 32});
num::Vector x;
F.solve(b, x);
F.solve_transpose(b, x);
```

## Inverse Entries Without Forming the Full Inverse

```cpp
auto F = num::lu(A_dense);
num::InverseDiagonalWorkspace work;

num::Vector inverse_diagonal(A_dense.rows(), 0.0);
num::inverse_diagonal(F, inverse_diagonal, work); // Blocked AX=I solves.
```

Request arbitrary entries or a principal block:

```cpp
std::array<num::idx, 2> rows{0, 2};
std::array<num::idx, 2> cols{1, 2};
num::Vector entries(2, 0.0);
num::selected_inverse(F, rows, cols, entries, work);

std::array<num::idx, 2> indices{0, 2};
num::Matrix principal;
num::inverse_principal_block(F, indices, principal, work);
```

After a low-rank change \f$A\leftarrow A+UWU^T\f$, reuse the old inverse
diagonal through Woodbury when numerically safe:

```cpp
num::Matrix U(A.rows(), 1, 0.0);
U(0, 0) = 1.0;
num::Matrix W(1, 1, 0.25);

num::Matrix updated = A;
updated(0, 0) += 0.25;
auto updated_F = num::lu(updated);

num::Vector updated_diagonal(A.rows(), 0.0);
auto path = num::inverse_diagonal_after_update(
    F, updated_F, inverse_diagonal, U, W,
    updated_diagonal, work);
```

`path` reports `woodbury` or `direct`. The updated factor is always supplied so
the routine can fall back to blocked identity solves after singular reduced
systems, inaccurate small solves, or cancellation.

## CG, PCG, MINRES, and GMRES

```cpp
num::operators::SparseOp Aop(A_sparse);
num::Vector x(A_sparse.n_rows(), 0.0);

auto spd = num::operators::assume_spd(Aop);
auto cg_info = num::cg(spd, b, x, 1e-10, 1000);

auto M = num::jacobi_preconditioner(A_sparse);
auto pcg_info = num::pcg(spd, M, b, x, 1e-10, 1000);

auto symmetric = num::operators::assume_symmetric(Aop);
auto minres_info = num::minres(symmetric, b, x, 1e-10, 1000);

auto gmres_info = num::gmres(Aop, b, x, 1e-8, 1000, 40);
```

The high-level facade keeps the algorithm choice explicit:

```cpp
auto problem = num::LinearProblem{spd, b};
auto result = num::solve(problem, num::CG{.tol = 1e-10});

auto cache = num::init(problem, num::CG{}, result.u);
auto repeated = num::solve(cache); // Warm-start from cache.u.
```

Every iterative routine returns iterations, final residual norm, and a
convergence flag.

## Symmetric Eigenvalues

```cpp
auto full = num::eig_sym(A); // All eigenpairs of symmetric A.

auto dominant = num::power_iteration(A);
auto near_shift = num::inverse_iteration(A, 2.0);
auto refined = num::rayleigh_iteration(A, dominant.eigenvector);

num::operators::DenseOp Aop(A);
auto largest_four = num::lanczos(
    num::operators::assume_symmetric(Aop), 4);
```

`lanczos` also accepts dense and sparse stored matrices. Its result contains
Ritz values, Ritz vectors, performed steps, and a residual-based convergence
flag.

## Singular Value Decomposition

```cpp
auto full = num::svd(A);              // A = U*diag(S)*Vt
auto rank_eight = num::svd_truncated(A, 8); // Randomized truncated SVD

num::Rng rng(1234);
auto repeatable = num::svd_truncated(A, 8, num::blas, 10, &rng);
```

## Shifted Resolvents

```cpp
num::cplx shift{1.0, 2.0};
auto x = num::resolvent_solve(shift, A, b); // (shift*I-A)^-1 b

num::ResolventFactor F(shift, A);
std::vector<num::cplx> complex_b(b.size(), 1.0);
auto reused = F.solve(complex_b);

std::vector<num::cplx> shifts{{1.0, 0.0}, {1.0, 1.0}};
auto batch = num::resolvent_solve_batch(shifts, A, b);
```

For sparse shifted systems:

```cpp
num::AutoResolventSolver F(A_sparse, {.dense_limit = 128});
F.factorize(shift);

std::vector<num::cplx> x;
F.solve(complex_b, x);
```

The automatic solver uses dense LU below the threshold and sparse complex
UMFPACK above it.

## Matrix Exponential Actions

```cpp
num::operators::DenseOp Aop(A);
num::Vector y = num::expv(0.01, Aop, b, 30, 1e-8);

num::Vector sparse_y = num::expv(0.01, A_sparse, b, 30, 1e-8);
```

`expv` forms \f$\exp(tA)v\f$ by Arnoldi projection without materializing the
matrix exponential.

The executable demonstrations are also available in the generated examples
index:

@example 01_direct_factorizations.cpp
@example 02_iterative_krylov_solvers.cpp
@example 03_resolvent_and_expv.cpp
@example 04_eigen_and_svd.cpp
@example 10_banded_and_spd_operators.cpp
