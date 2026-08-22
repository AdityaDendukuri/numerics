# Operator Examples and Concepts {#page_operators}

Linear operators supply the matrix-free operation

\f[
    y \leftarrow A x .
\f]

They decouple algorithms (CG, GMRES, Lanczos, and exponential-action routines) from concrete storage layouts.
Structural concepts describe the callable interface; property wrappers declare mathematical
contracts such as symmetry or positive definiteness.

---

## The Operator Concept Taxonomy

Operator contracts are defined in `include/operator/concepts.hpp`:

```cpp
template <class Op, class X = Vector, class Y = Vector>
concept LinearOperator =
    VectorLike<X> && MutableVectorLike<Y> &&
    requires(const Op &A, const X &x, Y &y) {
        { A.rows() } -> std::convertible_to<idx>;
        { A.cols() } -> std::convertible_to<idx>;
        { A.apply(x, y) };
    };

template <class Op, class X = Vector, class Y = Vector>
concept AdjointableLinearOperator =
    LinearOperator<Op, X, Y> &&
    requires(const Op &A, const Y &y, X &x) {
        { A.apply_adjoint(y, x) };
    };
```

---

## Declared Mathematical Properties

Iterative solvers like Conjugate Gradient require mathematical properties that cannot be fully verified at compile time (such as positive definiteness). Numerics uses **property tags** paired with **runtime diagnostic sampling**:

```cpp
// 1. Tagging symmetry:
auto sym_op = num::operators::assume_symmetric(Aop);
static_assert(num::SymmetricLinearOperator<decltype(sym_op)>);

// 2. Tagging SPD:
auto spd_op = num::operators::assume_spd(Aop);
static_assert(num::SPDLinearOperator<decltype(spd_op)>);
```

When `num::debug::set_level(DiagnosticLevel::full)` is enabled (default), `assume_spd()` evaluates a sampled inner product $x^T A x > 0$. If the operator violates positive definiteness:

```text
[PropertyError] Error at include/operator/properties.hpp:67 in assume_spd:
  assume_spd() assertion failed: sampled inner product x^T A x = -2.400000 <= 0.
  The operator is NOT positive definite!
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
