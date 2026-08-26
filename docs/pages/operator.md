# Operator Examples & Matrix-Free Workflows {#page_operator}

Linear operators supply the matrix-free action $y \leftarrow A x$, decoupling Krylov iterative solvers (CG, GMRES, BiCGSTAB, MINRES, Lanczos) from concrete storage layouts.

For the theoretical concept taxonomy, mathematical property tags, 2-tier warning system, and full operator file tree, see @ref page_concepts "Concepts, Invariants & Diagnostics Architecture".

---

## Declared Mathematical Properties

Iterative solvers like Conjugate Gradient require mathematical properties that cannot be fully verified at compile time (such as positive definiteness). Numerics uses **property tags** paired with **runtime diagnostic sampling**:

```cpp
// 1. Tagging symmetry:
auto sym_op = num::operators::assume_symmetric(Aop);
static_assert(num::SelfAdjointOperator<decltype(sym_op)>);

// 2. Tagging SPD:
auto spd_op = num::operators::assume_spd(Aop);
static_assert(num::SPDOperator<decltype(spd_op)>);
```

---

## Dense Operator Example

```cpp
num::Matrix A(3, 3, 0.0);
A(0, 0) = 4.0; A(0, 1) = 1.0; A(0, 2) = 0.0;
A(1, 0) = 1.0; A(1, 1) = 4.0; A(1, 2) = 1.0;
A(2, 0) = 0.0; A(2, 1) = 1.0; A(2, 2) = 4.0;

num::operators::DenseOp Aop(A); // Non-owning reference
static_assert(num::LinearOperator<decltype(Aop)>);

num::Vector b{1.0, 2.0, 3.0};
num::Vector x(3, 0.0);

// Solves in 3 iterations
num::SolverResult info = num::cg(num::operators::assume_spd(Aop), b, x);
```

---

## Sparse CSR Operator Example

```cpp
num::SparseMatrix A = num::SparseMatrix::from_triplets(
    n, n, rows, cols, values);

// Validates CSR monotonic row_ptr, column bounds, and finite values:
num::debug::verify_sparse_structure(A);

num::operators::SparseOp Aop(A);
num::SolverResult info = num::gmres(Aop, b, x, 1e-10, 200);
```

---

## Matrix-Free Lambda / Stencil Operator

```cpp
// Create a matrix-free 5-point discrete Laplacian stencil:
auto Aop = num::operators::make_op(
    [N](const num::Vector &x, num::Vector &y) {
        apply_laplacian_stencil(x, y, N);
    },
    N * N);

static_assert(num::LinearOperator<decltype(Aop)>);

num::SolverResult info =
    num::cg(num::operators::assume_spd(Aop), b, x, 1e-8, 1000);
```

---

## Lanczos Spectral Truncation

```cpp
// Compute the top-k eigenvalues and eigenvectors:
auto eig = num::lanczos(num::operators::assume_symmetric(Aop), 20, 1e-10, 100);
```

---

## Krylov Matrix Exponential Propagation

```cpp
// Compute y = exp(t * A) * v in O(m * N) time without assembling exp(A):
num::Vector y = num::expv(t, Aop, v, 30, 1e-8);
```

Evaluates the matrix action without forming the dense \f$N \times N\f$ matrix exponential:

\f[
\mathbf{y} = \exp(t A) \mathbf{v} \approx \|\mathbf{v}\|_2 V_m \exp(t H_m) \mathbf{e}_1
\f]

