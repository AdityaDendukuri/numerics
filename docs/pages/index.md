# numerics {#mainpage}

`numerics` is a modern C++20 numerical computing library for scientific computing, physical simulation, and applied mathematics. It provides cache-aligned dense and sparse linear algebra, direct factorizations, Krylov iterative solvers, adaptive and symplectic ODE integrators, spectral FFT transforms, graph algorithms, and quadrature methods. Compute kernels are designed with zero hidden heap allocations in simulation loops and enforce mathematical preconditions (such as positive-definiteness or symmetry) through C++20 concepts and runtime diagnostic evidence.

This started off as a personal compilation of my research and coursework code into a unified applied math library. I develop this package alongside downstream projects, continuously absorbing and refining new numerical tools into `numerics` for re-use. Because this has primarily been built for my own research workflows rather than by a large team, please use it with appropriate caution!

Over time, this package has grown to include everything from my undergraduate mesh-free fluid solvers (developed for surgical simulation) and master's work on graph algorithms and Ising nucleation, to my PhD research on finite state projection and iterative linear solvers.

Despite its organic evolution, the library is built on modern C++20 with 321 unit tests, clean fallback paths (from BLAS/LAPACK/OpenMP/CUDA acceleration down to pure portable C++).

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
5. @subpage page_concepts "Concepts & Invariants" (all 83 concepts, then structure versus law, declaring them, and the diagnostics)
6. @subpage page_expressive "Expression Interface" (convenience operators and performance tradeoffs)
7. @subpage page_examples "Examples" (code organized by numerical domain)
8. @subpage page_reference "API Reference" (classes, functions, and concepts)
9. @subpage page_report "Benchmark Report" (kernel throughput, convergence, validation)
10. @subpage page_developer "Developer Documentation" (testing and contribution standards)

