# numerics {#mainpage}

`numerics` is a C++20 library for scientific computing.
It provides dense and sparse linear algebra, direct factorizations, Krylov solvers, ODE integrators, spectral transforms, graph algorithms, and quadrature.
The compute kernels are efficient, allocate nothing, and have no dependencies.
The layers above them state the mathematical preconditions of each algorithm, such as symmetry or positive-definiteness, as C++20 concepts.
A precondition that cannot be established at compile time is asserted by the caller and verified at run time under the diagnostic presets.

The library originated as a consolidation of my research and coursework code, and it continues to grow in that manner.
Tools developed for downstream projects are incorporated here and refined for reuse.
Its contents span mesh-free fluid solvers from undergraduate work on surgical simulation, graph algorithms and Ising nucleation from my master's thesis, and finite state projection and iterative linear solvers from my doctoral research.
It is maintained by one person in support of that research.
Please use it with appropriate caution!!

The library is covered by 366 unit tests.
BLAS, LAPACK, OpenMP, and CUDA accelerate it when they are available.
When they are not, portable C++ takes their place.
On one core, the portable dense product and factorizations run within 10% of a vendor BLAS, and the small triangular solves run faster than LAPACK at every size (see @ref page_performance "Performance").

Jump right in with @ref page_getting_started "Getting Started" or browse @ref page_examples "Examples".

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

auto spd_L = num::operators::assume_spd(laplacian);
num::vec u(N * N, 0.0);
num::cg(spd_L, rhs, u, 1e-8);
```

### 3. Unified Problem Dispatch
```cpp
#include <numerics.hpp>

auto op = num::operators::dense_op(A);
auto solution = num::solve(
    num::linear_problem{op, b},
    num::gmres_method{.tol = 1e-10, .max_iter = 200});
```

---

## Architecture and Design Rules

1. **Deterministic Allocation:** Raw compute kernels operate on caller-provided output buffers; no hidden allocations in simulation loops.
2. **Layered Modules:** `kernel` has zero dependencies, `core` and `algebra` define types and concepts, and domain modules build on both (see @ref page_architecture).
3. **Storage / Operator Decoupling:** Solvers accept anything implementing the required mathematical protocol (`vector_space`, `linear_operator`), whether stored as `mat` or evaluated on the fly via `make_op`.
4. **Hardware Acceleration:** The library compiles against standard C++20 alone. Each accelerator is a plain namespace, such as `num::omp::dot` or `num::blas::matmul`. Select one by name, or let the build's configuration decide. There is no tag or enum layer between the caller and the backend. See @ref page_parallel.
5. **Enforced Invariants:** Algorithms state required properties (`spd_operator`, `self_adjoint_operator`). Passing an uncertified type fails at compile time; runtime claims (`assume_spd`) are validated under diagnostic presets (see @ref page_concepts).

---

## Documentation

1. @subpage page_getting_started "Getting Started" (CMake setup, header inclusion, basic operations)
2. @subpage page_architecture "Library Structure & Architecture" (standalone raw compute layer, module tiers, dependency invariants, where a new feature goes)
3. @subpage page_kernel "num::kernel" (all 88 routines, then the contract, parameter tables, and vendoring)
4. @subpage page_parallel "Backend Namespaces & Hardware Acceleration" (switching between kernel, BLAS/LAPACK, OpenMP, CUDA, and MPI)
5. @subpage page_container "Containers" (all 81 types and routines, then the vocabulary, vectors, matrices, storage layout, and sparse)
6. @subpage page_concepts "Concepts & Invariants" (all 76 concepts, then structure versus law, declaring them, and the diagnostics)
7. @subpage page_expressive "Expression Interface" (convenience operators and performance tradeoffs)
8. @subpage page_examples "Examples" (code organized by numerical domain)
9. @subpage page_reference "API Reference" (classes, functions, and concepts)
10. @subpage page_report "Benchmark Report" (kernel throughput, convergence, validation)
11. @subpage page_developer "Developer Documentation" (testing and contribution standards)

