# Linear Algebra {#page_linear}

Direct factorizations, iterative Krylov solvers, eigenvalue/SVD routines, banded solvers, and graph matrix generators.

---

## 1. Direct Factorizations

### num::lu
Decomposes square matrix \f$A \in \mathbb{R}^{n \times n}\f$ into \f$P A = L U\f$ with partial pivoting.

```cpp
num::LUResult lu(const num::SquareMatrixLike auto& A,
                 auto backend = num::backend::factor);
```

* **Complexity:** \f$\mathcal{O}(n^3)\f$ time, \f$\mathcal{O}(n^2)\f$ space.
* **Preconditions:** `A.rows() == A.cols()`. Wrap with `num::assume_square(A)` if untagged.

```cpp
num::Matrix A(3, 3, 0.0);
// fill A...
num::LUResult factor = num::lu(num::assume_square(A));
if (factor.singular) {
    // A is singular to machine precision
}

num::Vector b{5.0, 6.0, 5.0};
num::Vector x(3, 0.0);

num::lu_solve(factor, b, x);           // Solves A * x = b in O(n^2)
num::lu_solve_transpose(factor, b, x); // Solves A^T * x = b in O(n^2)

// Multiple right-hand sides
num::Matrix B = num::identity_columns(3, 0, 2);
num::Matrix X(3, 2, 0.0);
num::lu_solve(factor, B, X);           // Solves A * X = B

num::real det = num::lu_det(factor);   // det(A)
num::Matrix inv = num::lu_inv(factor); // A^{-1}
```

---

### num::cholesky
Decomposes symmetric positive-definite (SPD) matrix \f$A \in \mathbb{R}^{n \times n}\f$ into \f$A = L L^T\f$.

```cpp
num::CholeskyResult cholesky(const num::SPDMatrixLike auto& A,
                             auto backend = num::backend::factor);
```

* **Complexity:** \f$\mathcal{O}(n^3/3)\f$ time, \f$\mathcal{O}(n^2)\f$ space.
* **Preconditions:** \f$A = A^T\f$ and \f$x^T A x > 0\f$ for \f$x \ne 0\f$. Wrap with `num::assume_spd(A)` or `num::make_spd(A)`.

```cpp
num::Matrix A(2, 2, 0.0);
A(0, 0) = 4.0; A(0, 1) = 1.0;
A(1, 0) = 1.0; A(1, 1) = 3.0;

auto spd = num::assume_spd(A);
num::CholeskyResult factor = num::cholesky(spd);

num::Vector b{1.0, 2.0};
num::Vector x(2, 0.0);
num::cholesky_solve(factor, b, x); // Solves A * x = b

// Rank-1 updates: A_new = A +/- u * u^T
num::Vector u{0.1, -0.1};
num::cholesky_update(factor, u);   // O(n^2) update for A + u*u^T
num::cholesky_downdate(factor, u); // O(n^2) downdate for A - u*u^T
```

---

### num::qr
Decomposes \f$A \in \mathbb{R}^{m \times n}\f$ (\f$m \ge n\f$) into \f$A = Q R\f$ via Householder reflectors.

```cpp
num::QRResult qr(const num::DenseMatrixLike auto& A);
```

* **Complexity:** \f$\mathcal{O}(m n^2)\f$ time.

```cpp
num::Matrix A(4, 2, 0.0);
// fill observations...
num::QRResult factor = num::qr(A);

num::Vector b{1.0, 2.0, 2.0, 4.0};
num::Vector x(2, 0.0);
num::qr_solve(factor, b, x); // Minimizes ||A*x - b||_2
```

---

### num::HessenbergResolventSolver
Reduces square \f$A\f$ to upper Hessenberg form \f$A = Q H Q^T\f$ in \f$\mathcal{O}(n^3)\f$, then solves shifted systems \f$(z_k I - A) x_k = b\f$ in \f$\mathcal{O}(n^2)\f$ per shift.

```cpp
num::HessenbergResolventSolver solver(A); // Computes A = Q*H*Q^T in O(n^3)

num::cplx z{1.0, 2.0};
std::vector<num::cplx> rhs(A.rows(), num::cplx{1.0, 0.0});
std::vector<num::cplx> x = solver.solve(z, rhs); // O(n^2) shifted solve
```

For manual buffer reuse across shifts:
```cpp
num::HessenbergDecomposition decomp(A);
std::vector<num::cplx> work, y, x;
std::vector<num::idx> pivots;

auto b_tilde = num::hessenberg_project(decomp.Q(), b); // b_tilde = Q^T * b
num::hessenberg_shifted_solve(decomp.H(), z, b_tilde, y, work, pivots);
num::hessenberg_back_project(decomp.Q(), y, x);        // x = Q * y
```

---

### num::thomas
Solves tridiagonal linear system \f$T x = d\f$ in \f$\mathcal{O}(n)\f$ time and \f$\mathcal{O}(1)\f$ auxiliary storage.

```cpp
void thomas(const num::Vector& lower,
            const num::Vector& diag,
            const num::Vector& upper,
            const num::Vector& rhs,
            num::Vector& x);
```

```cpp
num::Vector lower{-1.0, -1.0};
num::Vector diag{2.0, 2.0, 2.0};
num::Vector upper{-1.0, -1.0};
num::Vector rhs{1.0, 0.0, 1.0};
num::Vector x(3, 0.0);

num::thomas(lower, diag, upper, rhs, x); // O(n) tridiagonal solve
```

---

### num::banded_solve
Solves banded linear system \f$A x = b\f$ with lower bandwidth \f$k_l\f$ and upper bandwidth \f$k_u\f$ in \f$\mathcal{O}(n \cdot k_l \cdot k_u)\f$ time.

```cpp
num::BandedMatrix A(128, /*kl=*/1, /*ku=*/1);
for (num::idx i = 0; i < A.rows(); ++i) {
    A(i, i) = 2.0;
    if (i > 0) A(i, i - 1) = -1.0;
    if (i + 1 < A.rows()) A(i, i + 1) = -1.0;
}

num::Vector b(128, 1.0);
num::Vector x(128, 0.0);
num::banded_solve(A, b, x);
```

---

### Selected Inversion (Takahashi Algorithm)
Computes entries of \f$A^{-1}\f$ without materializing the full inverse.

```cpp
auto factor = num::lu(num::assume_square(A));
num::InverseDiagonalWorkspace work;

// 1. Diagonal entries diag(A^{-1})
num::Vector diag(A.rows(), 0.0);
num::inverse_diagonal(factor, diag, work);

// 2. Selected entries A^{-1}(rows[i], cols[i])
std::vector<num::idx> rows{0, 2}, cols{0, 2};
num::Vector entries(2, 0.0);
num::selected_inverse(factor, rows, cols, entries, work);

// 3. Principal k x k block [A^{-1}]_{S, S}
std::vector<num::idx> subset{0, 1};
num::Matrix block;
num::inverse_principal_block(factor, subset, block, work);
```

---

## 2. Iterative Krylov Solvers

| Routine | Matrix / Operator Requirement | Minimization Guarantee |
| :--- | :--- | :--- |
| `num::cg` | Symmetric Positive Definite (\f$A \succ 0\f$) | Minimizes \f$\Vert x_k - x^* \Vert_A\f$ |
| `num::pcg` | SPD Operator & SPD Preconditioner (\f$A, M \succ 0\f$) | Minimizes \f$\Vert x_k - x^* \Vert_A\f$ |
| `num::minres` | Symmetric Indefinite (\f$A = A^T\f$) | Minimizes Euclidean residual \f$\Vert r_k \Vert_2\f$ |
| `num::gmres` | General Nonsymmetric | Minimizes \f$\Vert r_k \Vert_2\f$ over Arnoldi basis |
| `num::bicgstab` | General Nonsymmetric | Bi-Conjugate Gradient Stabilized |

### Signatures
```cpp
SolverResult cg(const SPDOperator auto& A, const auto& b, auto& x,
                real tol = 1e-8, idx max_iter = 1000);

SolverResult pcg(const SPDOperator auto& A, const auto& M, const auto& b, auto& x,
                 real tol = 1e-8, idx max_iter = 1000);

SolverResult minres(const SelfAdjointOperator auto& A, const auto& b, auto& x,
                    real tol = 1e-8, idx max_iter = 1000);

SolverResult gmres(const LinearOperator auto& A, const auto& b, auto& x,
                   real tol = 1e-8, idx max_iter = 1000, idx restart = 30);

SolverResult bicgstab(const LinearOperator auto& A, const auto& b, auto& x,
                      real tol = 1e-8, idx max_iter = 1000);
```

### Usage
```cpp
num::operators::SparseOp Aop(A_sparse);

// CG (requires SPD invariant)
auto spd_A = num::assume<num::axiom::positive_definite>(Aop);
num::Vector x_cg(n, 0.0);
num::SolverResult r_cg = num::cg(spd_A, b, x_cg, 1e-10, 500);

// PCG with Jacobi preconditioner
auto M = num::jacobi_preconditioner(A_sparse);
num::SolverResult r_pcg = num::pcg(spd_A, M, b, x_cg, 1e-10, 500);

// MINRES (requires symmetric / self-adjoint invariant)
auto sym_A = num::assume<num::axiom::self_adjoint>(Aop);
num::Vector x_minres(n, 0.0);
num::SolverResult r_minres = num::minres(sym_A, b, x_minres, 1e-10, 500);

// GMRES (requires only linearity)
num::Vector x_gmres(n, 0.0);
num::SolverResult r_gmres = num::gmres(Aop, b, x_gmres, 1e-8, 500, /*restart=*/30);
```

### Restricted-Subspace PCG (Singular Graph Laplacians)
For systems like graph Laplacians where \f$L \succeq 0\f$ is singular with nullspace \f$\mathbf{1}\f$, PCG is restricted to the compatible zero-sum subspace \f$S = \{v : \mathbf{1}^T v = 0\}\f$:

```cpp
num::space::zero_sum S;
auto restricted_L = num::assume<num::axiom::positive_definite_on<num::space::zero_sum>>(Lop);
auto projected_M  = num::operators::projected(M, S);
auto restricted_M = num::assume<num::axiom::positive_definite_on<num::space::zero_sum>>(projected_M);

num::SolverResult res = num::pcg(restricted_L, restricted_M, b, x, S,
                                 {.tolerance = 1e-10, .max_iterations = 1000});
```

---

## 3. Eigenvalues and SVD

### num::eig_sym
Computes all eigenvalues \f$\lambda_i\f$ and eigenvectors \f$v_i\f$ of symmetric \f$A = A^T\f$ via cyclic Jacobi rotations.

```cpp
num::EigenResult eig_sym(const num::SymmetricMatrixLike auto& A);
```

```cpp
num::EigenResult res = num::eig_sym(num::assume_symmetric(A));
num::Vector eigenvalues  = res.values;  // length n
num::Matrix eigenvectors = res.vectors; // n x n columns
```

### num::lanczos
Computes top-\f$k\f$ extremal eigenvalues and Ritz vectors of a symmetric operator via Lanczos iteration.

```cpp
num::operators::DenseOp Aop(A);
auto sym = num::operators::assume_symmetric(Aop);
num::LanczosResult res = num::lanczos(sym, /*k=*/4, /*tol=*/1e-10, /*max_iter=*/100);
```

### num::svd & num::svd_truncated
Computes Singular Value Decomposition \f$A = U \Sigma V^T\f$.

```cpp
num::SVDResult res = num::svd(A);                      // Exact full SVD
num::SVDResult r8  = num::svd_truncated(A, /*rank=*/8);// Randomized SVD for top 8 pairs
```

---

## 4. Matrix Exponential (num::expv)

Computes \f$y = \exp(t A) b\f$ via Krylov subspace projection without forming \f$\exp(t A)\f$.

```cpp
num::operators::DenseOp Aop(A);
num::Vector y = num::expv(/*t=*/0.01, Aop, b, /*krylov_subspace_size=*/30, /*tol=*/1e-8);
```

---

## 5. Graph Matrices

Assembles combinatorial graph Laplacian \f$L = D - A\f$ and continuous-time Markov generator \f$Q = -L\f$.

```cpp
// Sparse CSR
num::SparseMatrix L_sp = num::linear::laplacian(G);
num::SparseMatrix Q_sp = num::linear::markov_generator(G, /*column_oriented=*/true);

// Dense Matrix
num::Matrix L_dense = num::linear::dense_laplacian(G);
num::Matrix Q_dense = num::linear::dense_markov_generator(G, /*column_oriented=*/true);
```

---

## 6. Value-Returning Expression Interface

For rapid mathematical prototyping, concise unit testing, and formula readability, Numerics provides an opt-in convenience expression tier:

### Matrix & Vector Constructors
```cpp
num::Matrix Z = num::zeros(4, 4);      // 4x4 zero matrix
num::Matrix O = num::ones(3, 2);       // 3x2 matrix filled with 1.0
num::Matrix I = num::eye(4);           // 4x4 identity matrix
num::Vector v = num::linspace(0, 1, 5);// Vector: [0.0, 0.25, 0.5, 0.75, 1.0]
num::real s   = num::accu(Z);          // Sum of all elements
```

### Infix Operator Overloads (num::ops)
```cpp
using namespace num::ops;

num::Matrix A = num::ones(3, 3);
num::Matrix B = num::eye(3);
num::Vector x{1.0, 2.0, 3.0};

// Natural algebraic expressions
num::Matrix C = A * B + 2.0 * B;
num::Vector y = A * x - x / 2.0;
```

---

### Why Value-Returning Expressions Are Not Preferable in Production Code

While value-returning infix expressions are convenient for scripting and high-level setup, they are **strongly discouraged inside performance-critical simulations, numerical ODE integrators, and inner solver loops** for the following architectural reasons:

1. **Hidden Dynamic Memory Allocations:**
   Every binary operator (`*`, `+`, `-`, `/`) allocates a new heap buffer to return its result. For example, evaluating `y = A * x + B * z - C * w` creates three intermediate vector allocations and three corresponding deallocations. Inside a time-stepping loop executing \f$10^6\f$ iterations, this generates millions of heap operations, leading to allocator lock contention and cache invalidation.

2. **Absence of Loop Fusion:**
   Without complex lazy expression template trees, each sub-expression must be evaluated into a temporary buffer in a separate memory pass before the next operator can execute. This significantly increases memory bandwidth pressure compared to a single fused loop.

3. **Bypassing Mathematical Invariants & Verification:**
   Convenience solvers (such as uncertified `solve(A, b)` or `inv(A)`) treat all matrices as general dense arrays. They cannot exploit structural guarantees—such as positive-definiteness, bandedness, or symmetry—that allow the library to select \f$\mathcal{O}(n)\f$ or \f$\mathcal{O}(n^2)\f$ algorithms.

### The Recommended Numerics Idiom: Zero-Allocation Kernels
In production code and simulation loops, always prefer pre-allocated buffers and out-parameter mutating kernels:

```cpp
// Allocate once outside the simulation loop
num::Vector y(n, 0.0);
num::Vector temp(n, 0.0);

for (num::idx step = 0; step < num_steps; ++step) {
    // Zero-allocation, hardware-accelerated kernels
    num::matvec(A, x, y);           // y = A * x
    num::axpy(2.0, z, y);           // y = y + 2.0 * z
    num::cholesky_solve(L, b, x);   // Solves L * L^T * x = b in place
}
```

See @ref page_expressive "Expression Interface" for complete constructor, reduction, operator reference, and benchmark details.

---

## Examples

- @example 01_direct_factorizations.cpp
- @example 02_iterative_krylov_solvers.cpp
- @example 03_resolvent_and_expv.cpp
- @example 04_eigen_and_svd.cpp
- @example 10_banded_and_spd_operators.cpp
- @example 14_concepts_and_property_invariants.cpp
- @example 15_diffusion_evidence_cg.cpp
