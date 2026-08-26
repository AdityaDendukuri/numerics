# numerics {#mainpage}

Numerics is an executable-mathematics library for C++20. Mathematical
structure determines which algorithms are valid, evidence-bearing values carry
runtime invariants, and generic operations lower to allocation-controlled raw
kernels.

Its public design separates executable protocols, explicitly certified
algebraic laws, propositions about runtime values, and representation-specific
execution. This lets ordinary calls stay short while keeping assumptions visible
where correctness depends on them.

- **Source Repository:** [github.com/AdityaDendukuri/numerics](https://github.com/AdityaDendukuri/numerics)

---

## Core Interfaces

### 1. Direct Factorization and Solve
```cpp
#include <numerics.hpp>

num::Matrix A(2, 2, 0.0);
A(0, 0) = 4.0; A(0, 1) = 1.0;
A(1, 0) = 1.0; A(1, 1) = 3.0;

num::Vector b{1.0, 2.0};
auto factor = num::cholesky(num::linear::make_spd(A));

num::Vector x(2, 0.0);
num::cholesky_solve(factor, b, x); // Solves A * x = b
```

### 2. High-Level Problem Dispatch
Linear systems and differential equations can be dispatched through unified problem descriptors:

```cpp
auto op = num::operators::DenseOp(A); // Non-owning operator view

auto solution = num::solve(
    num::LinearProblem{op, b},
    num::GMRES{.tol = 1e-10, .max_iter = 200});
```

### 3. Matrix-Free Differential Operators
Spatial stencils and linear maps can be evaluated without assembling global sparse matrices:

```cpp
// 5-point discrete Laplacian stencil on an N x N grid
auto laplacian = num::operators::make_op(
    [N](const num::Vector &u, num::Vector &Lu) {
        apply_fd_laplacian(u, Lu, N);
    }, N * N);

auto spd_laplacian = num::operators::assume_spd(laplacian);
num::Vector u(N * N, 0.0);
num::cg(spd_laplacian, rhs, u, 1e-8);
```

---

## Design Principles

* **Deterministic Allocation:** Numerical kernels operate on caller-provided output buffers to prevent dynamic allocation inside simulation loops.
* **Layered Modules:** `kernel` is raw compute with no dependencies, `algebra` supplies the structure that code is written against, and the numerical modules build on both (see @ref page_architecture "Library Layout").
* **Storage and Operator Decoupling:** Concrete storage containers (`Vector`, `Matrix`, `SparseMatrix`) are decoupled from operator abstractions (`DenseOp`, `SparseOp`, `CallableOp`).
* **Multi-Backend Execution:** Algorithmic logic compiles against standard C++20 with optional acceleration paths for BLAS/LAPACK, OpenMP, SIMD/NEON, FFTW3, SuiteSparse, MPI, and CUDA.
* **Mathematical Invariants:** A solver states the property it requires. Code that has not established that property does not compile. Properties that cannot be decided from a type are sampled at runtime under the active preset (see @ref page_concepts "Concepts & Invariants").

---

## Documentation Structure

Read in this order.

1. @subpage page_getting_started "Getting Started" (compilation, headers, and basic operations)
2. @subpage page_architecture "Library Layout" (what each module holds and when to reach for it)
3. @subpage page_concepts "Concepts & Invariants" (algebraic structure, property tags, presets, diagnostics)
4. @subpage page_examples "Examples" (code organized by numerical domain)
5. @subpage page_guides "Guides" (solver selection, backends, performance)
6. @subpage page_reference "API Reference" (classes, functions, and concepts)
7. @subpage page_report "Benchmark Report" (kernel throughput, convergence, validation)
8. @subpage page_developer "Developer Documentation" (testing and contribution standards)
- [Interactive Benchmark Report](report/index.html) (self-contained HTML report with embedded figures)
- [GitHub Repository](https://github.com/AdityaDendukuri/numerics) (source code, issues, and releases)
