# Library Layout {#page_architecture}

Numerics is organized in layers. Each layer depends only on the ones above it in the table below, so a module can be read, tested, or copied without pulling in the rest.

---

## 1. The Layers

| Layer | Holds | Depends on |
| :--- | :--- | :--- |
| `kernel` | Raw compute over pointers and callables | nothing |
| `core` | `idx`, `real`, `cplx`, backend policy, diagnostics | nothing |
| `algebra` | Scalar fields, vector spaces, property hierarchy | `core`, `kernel` |
| `structures` | Discrete structures and their algorithms | `core`, `algebra` |
| `container` | `Vector`, `Matrix`, backend-dispatched operations | `core`, `algebra`, `kernel` |
| `operator` | Matrix-free linear maps and property wrappers | `algebra`, `container` |
| `linear` | Factorizations, solvers, eigenproblems, sparse | the above |
| `quadrature` | Composite, Gaussian, adaptive, and contour rules | `algebra`, `core` |
| `roots` | Bracketing and derivative-based root finding | `algebra`, `core` |
| `ode`, `pde`, `spectral`, `fields`, `spatial`, `stochastic`, `stats` | Numerical domains | the above |
| `solve` | Unified problem dispatch | everything |
| `io`, `plot`, `viz` | Optional bindings | depended on by nothing |

---

## 2. Which Layer to Reach For

### You want an algorithm and nothing else

Use `kernel`. It operates on `T *` and a dimension, allocates nothing, and includes nothing outside the standard library:

```cpp
#include "kernel/krylov.hpp"

std::vector<double> b(n, 1.0), x(n, 0.0), work(3 * n);

auto A = [&](const double *v, double *out) { /* y <- A v */ };
auto r = num::kernel::raw::cg(A, x.data(), b.data(), n, work.data(), 1e-12, 500);
```

The operator is a callable and the storage is yours, so a project can vendor these headers without adopting anything else in the library. `tests/test_kernel_standalone.cpp` compiles against a copy of `include/kernel` alone and links no library.

### You want the library's containers and solvers

Use `linear` with `container`:

```cpp
#include <numerics.hpp>

num::Matrix A = make_spd_matrix();
num::Vector b{1.0, 2.0, 3.0}, x(3, 0.0);

auto factor = num::cholesky(num::assume_spd(A)); // Computes A = L*L^T in O(n^3/3).
num::cholesky_solve(factor, b, x);
```

### You want to solve without forming a matrix

Use `operator`. A type with `rows`, `cols`, and `apply` is a linear map:

```cpp
struct Laplacian1D {
    using properties = num::property::spd;
    num::idx n;

    [[nodiscard]] num::idx rows() const { return n; }
    [[nodiscard]] num::idx cols() const { return n; }

    void apply(const num::Vector &x, num::Vector &y) const {
        for (num::idx i = 0; i < n; ++i) {
            y[i] = 2.0 * x[i] - (i > 0 ? x[i - 1] : 0.0) - (i + 1 < n ? x[i + 1] : 0.0);
        }
    }
};

num::cg(Laplacian1D{n}, b, x); // No matrix is assembled.
```

### You want to write code that works for any vector type

Use `algebra`. Constrain on the structure a routine needs:

```cpp
template <num::VectorSpace V>
void normalize(V &v) {
    num::algebra::scale_inplace(v, num::scalar_t<V>(1) / num::algebra::norm_of(v));
}
```

This accepts `num::Vector`, `num::CVector`, and `std::vector<float>` alike, and rejects `std::span<const double>` because a view has nowhere to hold a sum.

### You want graphs, union-find, or traversal

Use `structures`. Graphs here carry no algebraic operations. Laplacians and adjacency matrices are produced by `linear/graph`, because they return matrices.

---

## 3. What core Holds

`core` is the small base every layer reads, and holds the three things that are neither algebra nor a container.

```cpp
num::idx    // Index type.
num::real   // Default scalar.
num::cplx   // Complex scalar.

num::backend::seq     // Backend tags, resolved at compile time.
num::backend::blas
num::with_backend(b, [&](auto tag) { /* runtime value becomes a tag once */ });

num::set_preset(num::preset::strict); // Diagnostic level.
```

---

## 4. Where Concepts Live

The module that introduces an abstraction defines its concept. There is no separate tier of core concepts.

| Module | Introduces | Concepts |
| :--- | :--- | :--- |
| `algebra` | Scalar fields and vector spaces | `Field`, `VectorSpace`, `InnerProductSpace`, `MatrixSpace` |
| `container` | Storage layouts | `repr::Contiguous`, `repr::DenseRowMajor`, `repr::CSR`, `repr::Banded` |
| `operator` | Linear maps | `LinearOperator`, `SelfAdjointOperator`, `SPDOperator`, `UnitaryOperator` |
| `linear` | Factorizations and solvers | `SPDMatrixLike`, `TriangularFactor`, `Preconditioner` |
| `structures` | Discrete structures | `EquivalenceRelation`, `IncidenceStructure`, `AddressablePriorityQueue` |
| `ode` | Initial value problems | `IsODEProblem`, `IsSymplecticODEProblem`, `IsODEStepper` |

A module concept is a conjunction containing the concepts it builds on:

```cpp
// algebra/concepts.hpp
concept VectorSpace = AdditiveGroup<V, T> && requires(V &v, const V &x, T a) {
    algebra::scale_inplace(v, a);
    algebra::axpy_into(a, x, v);
};

// operator/concepts.hpp
concept LinearOperator =
    VectorSpace<X> && MutableVectorSpace<Y> && requires(const Op &A, const X &x, Y &y) {
        { A.rows() } -> std::convertible_to<num::idx>;
        { A.cols() } -> std::convertible_to<num::idx>;
        { A.apply(x, y) };
    };
```

A module takes from `algebra` only what it needs. `structures` uses `idx` and `Field` for edge weights. A union-find is not a vector space, so `EquivalenceRelation` builds on nothing algebraic.

Each module also owns a `debug.hpp` when it has diagnostics only it can express. `linear` verifies the structure of stored matrices, `ode` measures order of accuracy and symplecticity, `structures` checks the equivalence-relation axioms and the handshake lemma. Diagnostics that apply to anything, such as property sampling on a linear map, live in `algebra`.

---

## 5. Invariants

A change to the tree keeps all of these true.

1. `kernel` includes nothing outside the standard library.
2. `core` depends on nothing.
3. No module depends on a module below it in section 1.
4. Every concept has exactly one definition site.
5. Every algorithm has one implementation. Entry points on operators, on stored matrices, and on raw pointers forward to it.
6. `io`, `plot`, and `viz` are depended on by nothing.

Invariant 5 is why `num::cg`, `num::unsafe::cg`, and `num::kernel::raw::cg` share a body rather than repeating the iteration three times.
