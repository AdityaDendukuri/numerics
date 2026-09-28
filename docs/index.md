# numerics

`numerics` is a C++20 scientific computing library developed from research and coursework in numerical methods. It combines dependency-free, allocation-conscious kernels with higher-level interfaces that express mathematical requirements such as symmetry and positive-definiteness through C++ concepts and runtime diagnostics. The project collects reusable work from fluid simulation, stochastic modeling, graph algorithms, and numerical linear algebra, and is maintained by one person in support of ongoing research.

The library is covered by unit tests, run in CI with and without BLAS and LAPACK.
BLAS, LAPACK, OpenMP, and CUDA accelerate it when they are available.
When they are not, portable C++ takes their place.
On one core, the portable dense product and factorizations run within 10% of a vendor BLAS, and the small triangular solves run faster than LAPACK at every size (see [Performance](performance.md)).

Jump right in with [Getting Started](getting-started.md) or browse [Examples](reference/examples/index.md).

---

## Core Interfaces

### 1. Direct Factorization and Solve
```cpp
#include <numerics.hpp>

num::mat A(2, 2, 0.0);
A(0, 0) = 4.0; A(0, 1) = 1.0;
A(1, 0) = 1.0; A(1, 1) = 3.0;

num::vec b{1.0, 2.0};
num::vec x(2, 0.0);

auto factor = num::cholesky(num::assume_spd(A));
num::cholesky_solve(factor, b, x); // Solves A * x = b
```

### 2. Matrix-Free Iterative Solvers
```cpp
#include <numerics.hpp>

// 5-point discrete Laplacian stencil on an N x N grid
auto laplacian = num::operators::make_op(
    [N](const num::vec &u, num::vec &Lu) {
        apply_fd_laplacian(u, Lu, N);
    }, N * N);

auto spd_L = num::assume_spd(laplacian);
num::vec u(N * N, 0.0);
num::cg(spd_L, rhs, u, {.tolerance = 1e-8});
```

---

## Architecture and Design Rules

1. **Deterministic Allocation:** Raw compute kernels operate on caller-provided output buffers; no hidden allocations in simulation loops.
2. **Layered Modules:** `kernel` has zero dependencies, `core` and `algebra` define types and concepts, and domain modules build on both (see [Library Structure & Architecture](architecture.md)).
3. **Storage / Operator Decoupling:** Solvers accept anything implementing the required mathematical protocol (`vector_space`, `linear_operator`), whether stored as `mat` or evaluated on the fly via `make_op`.
4. **Hardware Acceleration:** The library compiles against standard C++20 alone. Each accelerator is a plain namespace, such as `num::omp::dot` or `num::blas::matmul`. Select one by name, or let the build's configuration decide. There is no tag or enum layer between the caller and the backend. See [Backend Namespaces, Parallelism, & Hardware Acceleration](backends.md).
5. **Enforced Invariants:** Algorithms state required properties (`spd_operator`, `self_adjoint_operator`). Passing a type without the law fails at compile time; runtime claims (`assume_spd`) are validated under diagnostic presets (see [Concepts, Laws & Diagnostics](concepts.md)).

---

## Documentation

- [Getting Started](getting-started.md): CMake setup, headers, and first programs.
- [Concepts, Laws & Diagnostics](concepts.md): what each algorithm requires, and how a type states it.
- [Architecture](architecture.md): the module tiers and their dependency rules.
- [Backends](backends.md): BLAS, LAPACK, OpenMP, CUDA and MPI, and how a call picks one.
- [Performance](performance.md): benchmarks and the kernel's design.
- [Algorithm notes](notes.md): derivations and measurements behind specific routines.
- [Reference](reference/index.md): one page per public name, generated from the headers.
- [Examples](reference/examples/index.md): complete programs.
- [Benchmark report](report/REPORT.md).
