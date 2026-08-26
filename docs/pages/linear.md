# Linear Algebra {#page_linear}

The `linear` module provides direct factorizations, iterative Krylov solvers, eigenvalue/SVD routines, sparse matrix solvers, shifted resolvents, and matrix exponential approximations.

---

## 1. LU Factorization (Gaussian Elimination with Partial Pivoting)

Decomposes a general matrix \f$A \in \mathbb{R}^{n \times n}\f$ into:

\f[
P A = L U
\f]

where \f$P\f$ is a row permutation matrix, \f$L\f$ is unit lower triangular, and \f$U\f$ is upper triangular.

### Factor a Dense Matrix

```cpp
num::Matrix A = make_matrix();
num::LUResult factor = num::lu(num::assume_square(A)); // Computes P*A = L*U in O(n^3).

if (factor.singular) {
    // A is singular to machine precision.
}
```

### Solve Systems \f$A x = b\f$ and \f$A^T x = b\f$

\f[
L y = P b, \qquad U x = y
\f]

```cpp
num::Vector b{5.0, 6.0, 5.0};
num::Vector x;

num::lu_solve(factor, b, x);           // Solves A*x = b in O(n^2).
num::lu_solve_transpose(factor, b, x); // Solves A^T*x = b in O(n^2).
```

### Solve Multiple Right-Hand Sides \f$A X = B\f$

```cpp
num::Matrix B = num::identity_columns(A.rows(), 0, 2);
num::Matrix X;

num::lu_solve(factor, B, X); // Solves A*X = B with a single LU factorization.
```

### In-Place Solve, Determinant, and Matrix Inversion

\f[
\det(A) = (-1)^{\text{sign}(P)} \prod_{i=1}^n U_{ii}, \qquad A^{-1} = U^{-1} L^{-1} P
\f]

```cpp
num::Vector x = b;
num::solve_in_place(factor, x); // Overwrites x with A^{-1}*b.

num::real determinant = num::lu_det(factor);
num::Matrix inverse = num::lu_inv(factor); // Computes A^{-1}.
```

---

## 2. Cholesky Factorization (\f$A = L L^T\f$)

For Symmetric Positive-Definite (SPD) matrices \f$A \succ 0\f$:

\f[
A = L L^T
\f]

where \f$L\f$ is lower triangular with strictly positive diagonal entries.

### Validate & Factor SPD Matrices

```cpp
auto spd = num::linear::make_spd(A);            // Validates A = A^T and positive definiteness.
num::CholeskyResult factor = num::cholesky(spd); // Computes A = L*L^T.
```

```cpp
auto spd = num::linear::assume_spd(A); // Skips runtime SPD check when invariant is known.
auto factor = num::cholesky(spd);
```

### Solve Systems \f$A x = b\f$ and \f$A X = B\f$

\f[
L y = b, \qquad L^T x = y
\f]

```cpp
num::Vector x;
num::cholesky_solve(factor, b, x); // Solves A*x = b.

num::Matrix X;
num::cholesky_solve(factor, B, X); // Solves A*X = B.
```

### Rank-1 Updates & Downdates

Updates the Cholesky factor \f$L\f$ after rank-1 perturbations \f$A \pm u u^T\f$ in \f$\mathcal{O}(n^2)\f$ time without refactorizing from scratch:

\f[
A_{\text{new}} = A \pm u u^T = L_{\text{new}} L_{\text{new}}^T
\f]

```cpp
num::Vector u{0.1, 0.0, -0.1};
num::cholesky_update(factor, u);   // Factors A + u*u^T in O(n^2).
num::cholesky_downdate(factor, u); // Factors A - u*u^T in O(n^2) (throws if indefinite).
```

---

## 3. QR Decomposition & Linear Least Squares

Factors \f$A \in \mathbb{R}^{m \times n}\f$ ($m \ge n$) using Householder reflectors:

\f[
A = Q R = \begin{bmatrix} Q_1 & Q_2 \end{bmatrix} \begin{bmatrix} R_1 \\ 0 \end{bmatrix}
\f]

where \f$Q \in \mathbb{R}^{m \times m}\f$ is orthogonal (\f$Q^T Q = I\f$) and \f$R \in \mathbb{R}^{m \times n}\f$ is upper triangular.

### Least-Squares Solution

Minimizes the Euclidean residual \f$\min_{x} \|A x - b\|_2\f$:

\f[
R_1 x = Q_1^T b
\f]

```cpp
num::Matrix A(4, 2, 0.0);
// Fill observations...

num::QRResult factor = num::qr(A); // A = Q*R.

num::Vector observations{1.0, 2.0, 2.0, 4.0};
num::Vector coefficients;
num::qr_solve(factor, observations, coefficients); // Minimizes ||A*x - b||_2.
```

---

## 4. Hessenberg Decomposition & Shifted Resolvents

Reduces a square matrix \f$A \in \mathbb{R}^{n \times n}\f$ to upper Hessenberg form (\f$H_{i,j} = 0\f$ for \f$i > j + 1\f$):

\f[
A = Q H Q^T
\f]

Accelerates multiple shifted resolvent solves \f$(z_k I - A)^{-1} b\f$ from \f$\mathcal{O}(K n^3)\f$ down to \f$\mathcal{O}(n^3 + K n^2)\f$:

\f[
(z_k I - A)^{-1} b = Q (z_k I - H)^{-1} Q^T b
\f]

```cpp
num::HessenbergResolventSolver solver(A); // Precomputes A = Q*H*Q^T in O(n^3).

num::cplx shift{1.0, 2.0};
std::vector<num::cplx> rhs(A.rows(), num::cplx{1.0, 0.0});

// Solves (shift*I - A)*x = rhs in O(n^2) by Gaussian elimination with partial pivoting.
std::vector<num::cplx> x = solver.solve(shift, rhs);
```

The three steps are also available on their own, for a caller that already holds
a Hessenberg factorization and wants to drive the shifts itself:

```cpp
num::HessenbergDecomposition decomp(A);

std::vector<num::cplx> work, y, x;
std::vector<num::idx> pivots;

auto b_tilde = num::hessenberg_project(decomp.Q(), b);  // b_tilde = Q^T * b
num::hessenberg_shifted_solve(decomp.H(), shift, b_tilde, y, work, pivots);
num::hessenberg_back_project(decomp.Q(), y, x);         // x = Q * y
```

`work` and `pivots` are scratch. Passing the same buffers back on the next shift
avoids reallocating.

---

## 5. Tridiagonal Systems (Thomas Algorithm)

Solves tridiagonal linear systems \f$T x = d\f$ in \f$\mathcal{O}(n)\f$ arithmetic operations and \f$\mathcal{O}(1)\f$ extra memory:

\f[
a_i x_{i-1} + b_i x_i + c_i x_{i+1} = d_i
\f]

```cpp
num::Vector lower{-1.0, -1.0};
num::Vector diagonal{2.0, 2.0, 2.0};
num::Vector upper{-1.0, -1.0};
num::Vector rhs{1.0, 0.0, 1.0};
num::Vector x;

num::thomas(lower, diagonal, upper, rhs, x); // O(n) Thomas algorithm.
```

---

## 6. Banded Matrix Solvers

Stores matrices with lower bandwidth \f$k_l\f$ and upper bandwidth \f$k_u\f$ in compact LAPACK band format.

\f[
A_{i,j} \ne 0 \implies -k_l \le j - i \le k_u
\f]

```cpp
num::BandedMatrix A(128, 1, 1); // 128 rows, kl = 1, ku = 1.

for (num::idx i = 0; i < A.rows(); ++i) {
    A(i, i) = 2.0;
    if (i > 0) A(i, i - 1) = -1.0;
    if (i + 1 < A.rows()) A(i, i + 1) = -1.0;
}

num::Vector b(A.rows(), 1.0);
num::Vector x;
num::BandedSolverResult result = num::banded_solve(A, b, x); // Solves in O(n * kl * ku).
```

---

## 7. Selected Inversion & Principal Inverse Blocks

### Selected Inversion (Takahashi Algorithm)

Computes specific entries of \f$A^{-1}\f$ (such as the inverse diagonal \f$\text{diag}(A^{-1})\f$, coordinate pairs, or principal submatrix blocks) without materializing the full dense inverse:

```cpp
auto factor = num::lu(num::assume_square(A));
num::InverseDiagonalWorkspace workspace;

// 1. Full inverse diagonal diag(A^{-1}) via blocked identity solves:
num::Vector diagonal(A.rows(), 0.0);
num::inverse_diagonal(factor, diagonal, workspace);

// 2. Selected entries A^{-1}(rows[i], cols[i]):
std::vector<num::idx> rows{0, 2};
std::vector<num::idx> cols{0, 2};
num::Vector entries(2, 0.0);
num::selected_inverse(factor, rows, cols, entries, workspace);

// 3. Principal k x k inverse submatrix [A^{-1}]_{S, S}:
std::vector<num::idx> subset{0, 1};
num::Matrix principal_block;
num::inverse_principal_block(factor, subset, principal_block, workspace);
```

---

## 8. Iterative Krylov Subspace Solvers

Iterative solvers approximate \f$x \in \mathcal{K}_m(A, b) = \text{span}\{b, A b, A^2 b, \dots, A^{m-1} b\}\f$.

| Solver | Mathematical Guarantee | Matrix Requirements |
| :--- | :--- | :--- |
| **`cg`** | Minimizes \f$\|x_k - x^*\|_A\f$ | Symmetric Positive Definite (\f$A \succ 0\f$) |
| **`pcg`** | Preconditioned CG: \f$M^{-1} A x = M^{-1} b\f$ | Symmetric Positive Definite (\f$A, M \succ 0\f$) |
| **`minres`** | Minimizes Euclidean residual \f$\|r_k\|_2\f$ | Symmetric Indefinite (\f$A = A^T\f$) |
| **`gmres`** | Minimizes residual \f$\|r_k\|_2\f$ over Arnoldi basis | General Nonsymmetric \f$A\f$ |
| **`bicgstab`** | Bi-Conjugate Gradient Stabilized | General Nonsymmetric \f$A\f$ |

```cpp
num::Matrix A = make_spd_matrix();
auto spd = num::require<num::axiom::positive_definite>(A); // Exhaustive validation.
num::Vector x(A.rows(), 0.0);

// Conjugate Gradients
num::SolverResult result =
    num::cg(spd, b, x, {.tolerance = 1e-10, .max_iterations = 1000});

// A construction whose derivation already proves SPD can attach an explicit
// assumption instead. The evidence records this origin and its source location.
num::operators::SparseOp Aop(A_sparse);
auto constructed_spd = num::assume<num::axiom::positive_definite>(Aop);
num::SolverResult constructed_result = num::cg(constructed_spd, b, x);

// Preconditioned Conjugate Gradients with Jacobi Preconditioner
auto M = num::jacobi_preconditioner(A_sparse);
auto spd_operator = num::assume<num::axiom::positive_definite>(Aop);
num::SolverResult result_pcg = num::pcg(
    spd_operator, M, b, x,
    {.tolerance = 1e-10, .max_iterations = 1000});

// MINRES accepts self-adjoint operators, including indefinite ones.
auto symmetric_operator = num::assume<num::axiom::self_adjoint>(Aop);
num::SolverResult result_minres = num::minres(
    symmetric_operator, b, x,
    {.tolerance = 1e-10, .max_iterations = 1000});

// GMRES requires only a certified linear endomorphism.
num::SolverResult result_gmres = num::gmres(
    Aop, b, x,
    {.tolerance = 1e-8, .max_iterations = 1000, .restart = 30});
```

The constraints follow the derivations: GMRES requires a linear endomorphism,
MINRES adds self-adjoint evidence, CG adds positive-definite evidence, and PCG
requires positive-definite evidence for both the operator and approximate
inverse. Raw matrices and unproved operators therefore cannot enter the stricter
methods accidentally. The explicit CG escape hatch remains `num::unsafe::cg`
for callers deliberately taking responsibility for the missing theorem.

Approximate Cholesky factors of graph Laplacians are positive semidefinite, not
globally positive definite: the constant vector remains in their nullspace. A
PCG solve restricted to the compatible zero-sum subspace must therefore attach
evidence for that exact restriction:

```cpp
num::space::zero_sum S;
auto restricted_L =
    num::assume<num::axiom::positive_definite_on<num::space::zero_sum>>(Lop);
auto projected_M = num::operators::projected(M, S); // Fix the factor's arbitrary gauge.
auto restricted_M =
    num::assume<num::axiom::positive_definite_on<num::space::zero_sum>>(projected_M);

auto result = num::pcg(restricted_L, restricted_M, b, x, S,
                       {.tolerance = 1e-10, .max_iterations = 1000});

// The same invariant reaches the problem-level solve interface.
auto solution = num::solve(num::LinearProblem{restricted_L, b},
                           num::PCGOn{restricted_M, S});
```

This evidence does not imply global positive definiteness. PCG checks that `b`
and the initial `x` belong to `S`, then checks that operator applications,
preconditioner applications, residuals, directions, and iterates remain in `S`.
A false invariance claim therefore fails at the point where the recurrence would
leave the certified space.

---

## 9. Eigenvalue & Singular Value Decompositions

### Symmetric Eigendecomposition (\f$A = V \Lambda V^T\f$)

Computes all eigenvalues \f$\lambda_i\f$ and orthonormal eigenvectors \f$v_i\f$ using cyclic Jacobi rotations:

\f[
A v_i = \lambda_i v_i, \qquad V^T V = I
\f]

```cpp
num::EigenResult result = num::eig_sym(num::assume_symmetric(A));
num::Vector eigenvalues = result.values;
num::Matrix eigenvectors = result.vectors;
```

### Lanczos Iteration (Top-\f$k\f$ Extremal Eigenvalues)

Constructs an orthonormal tridiagonal Krylov basis \f$A V_k = V_k T_k + \beta_k v_{k+1} e_k^T\f$:

```cpp
num::operators::DenseOp Aop(A);
auto symmetric = num::operators::assume_symmetric(Aop);
num::LanczosResult largest = num::lanczos(symmetric, 4); // 4 extremal Ritz pairs.
```

### Singular Value Decomposition (\f$A = U \Sigma V^T\f$)

\f[
A = U \Sigma V^T = \sum_{i=1}^{\min(m,n)} \sigma_i u_i v_i^T
\f]

```cpp
num::SVDResult result = num::svd(A); // Exact full SVD.
num::SVDResult rank_eight = num::svd_truncated(A, 8); // Randomized SVD for top 8 singular pairs.
```

---

## 10. Matrix Exponential Actions (\f$y = \exp(t A) b\f$)

Evaluates the matrix exponential action without materializing the dense matrix exponential \f$\exp(t A)\f$, using Krylov subspace projection:

\f[
\exp(t A) b \approx \|b\|_2 V_m \exp(t H_m) e_1
\f]

```cpp
num::operators::DenseOp Aop(A);
num::Vector y = num::expv(0.01, Aop, b, /*krylov_subspace_size=*/30, /*tol=*/1e-8);
```

---

## 11. Mathematical Invariants & Concepts

All linear algebra routines enforce strict mathematical rules at compile time (`SquareMatrixLike`, `SymmetricMatrixLike`, `SPDMatrixLike`, `BandedMatrixLike`, `TridiagonalMatrixLike`, `SparseMatrixCSRLike`). Tagging matrices via `num::assume_spd(A)` or `num::make_spd(A)` activates the 100% silent, certified fast path.

For the comprehensive concept taxonomy, 2-tier warning system, complete solver coverage mapping, and module file tree, see @ref page_concepts "Concepts, Invariants & Diagnostics Architecture".

---

## Complete Examples

@example 01_direct_factorizations.cpp
@example 02_iterative_krylov_solvers.cpp
@example 03_resolvent_and_expv.cpp
@example 04_eigen_and_svd.cpp
@example 10_banded_and_spd_operators.cpp
@example 14_concepts_and_property_invariants.cpp
@example 15_diffusion_evidence_cg.cpp

---

## Graph Laplacian & Markov Generator Assembly

### Combinatorial Graph Laplacian (\f$L = D - A\f$)

The unnormalized Graph Laplacian of an undirected graph \f$G = (V, E, W)\f$ is defined as:

\f[
L = D - A, \qquad L_{ij} = \begin{cases}
d_i = \sum_{k} w_{ik} & \text{if } i = j, \\
-w_{ij} & \text{if } (i, j) \in E, \\
0 & \text{otherwise}
\end{cases}
\f]

The matrix \f$L\f$ is symmetric positive semidefinite (\f$L \succeq 0\f$) with nullspace spanned by the constant vector \f$\mathbf{1}\f$ for each connected component.

### Markov Jump Generator (\f$Q = -L\f$)

For continuous-time Markov jump processes on state space \f$V\f$, the master equation governs the probability vector \f$\mathbf{p}(t)\f$:

\f[
\frac{d\mathbf{p}(t)}{dt} = Q \mathbf{p}(t), \qquad \mathbf{1}^T Q = \mathbf{0}^T, \quad Q_{ij} \ge 0 \ (i \ne j)
\f]

```cpp
// Sparse CSR representations
num::SparseMatrix L_sparse = num::linear::laplacian(G);
num::SparseMatrix Q_sparse = num::linear::markov_generator(G, /*column_oriented=*/true);

// Dense matrix representations
num::Matrix L_dense = num::linear::dense_laplacian(G);
num::Matrix Q_dense = num::linear::dense_markov_generator(G, /*column_oriented=*/true);
```

---
