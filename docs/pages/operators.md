# Operator Examples and Concepts {#page_operators}

Linear operators supply the operation

\f[
    y \leftarrow A x .
\f]

They are used by CG, GMRES, Lanczos, and exponential-action routines. Structural
concepts describe the callable interface; property wrappers declare numerical
contracts such as symmetry or positive definiteness.

## Storage Concepts

Storage concepts are defined in `include/core/concepts.hpp`:

- `num::VectorLike`
- `num::MutableVectorLike`
- `num::ContiguousVectorLike`
- `num::DenseMatrixLike`
- `num::MutableDenseMatrixLike`
- `num::ContiguousDenseMatrixLike`

Operator concepts are defined in `include/operator/concepts.hpp`:

```cpp
template<class Op, class X = Vector, class Y = Vector>
concept LinearOperator =
    VectorLike<X> && MutableVectorLike<Y> &&
    requires(const Op& A, const X& x, Y& y) {
        { A.rows() } -> std::convertible_to<idx>;
        { A.cols() } -> std::convertible_to<idx>;
        { A.apply(x, y) };
};
```

Declared mathematical-property concepts:

- `num::SymmetricLinearOperator`
- `num::SPDLinearOperator`

These concepts are satisfied by wrappers such as `assume_symmetric(Aop)` and
`assume_spd(Aop)`. They do not prove the property; they make the numerical
contract explicit at the call site.

## Dense Operator

```cpp
num::Matrix A(3, 3, 0.0);
fill_spd_matrix(A);

num::operators::DenseOp Aop(A);
static_assert(num::LinearOperator<decltype(Aop)>);

num::Vector x(3, 0.0);
num::SolverResult info = num::cg(num::operators::assume_spd(Aop), b, x);
```

## Sparse Operator

```cpp
num::SparseMatrix A = num::SparseMatrix::from_triplets(
    n, n, rows, cols, values);

num::operators::SparseOp Aop(A);
num::SolverResult info = num::gmres(Aop, b, x, 1e-10, 200);
```

## Callable Operator

```cpp
auto Aop = num::operators::make_op(
    [N](const num::Vector& x, num::Vector& y) {
        apply_stencil(x, y, N);
    },
    N * N);

num::SolverResult info =
    num::cg(num::operators::assume_spd(Aop), b, x, 1e-8, 1000);
```

## Lanczos

```cpp
auto eig = num::lanczos(num::operators::assume_symmetric(Aop), 20, 1e-10, 100);
```

## Exponential Action

```cpp
num::Vector y = num::expv(t, Aop, v, 30, 1e-8);
```

Use operators for algorithms that only require products with \f$A\f$. Use direct
factorizations when the assembled matrix and repeated solves justify it.
