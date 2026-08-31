# numerics {#mainpage}

`numerics` is a modern C++20 numerical computing library for scientific computing, physical simulation, and applied mathematics. It provides cache-aligned dense and sparse linear algebra, direct factorizations, Krylov iterative solvers, adaptive and symplectic ODE integrators, spectral FFT transforms, graph algorithms, and quadrature methods. Compute kernels are designed with zero hidden heap allocations in simulation loops and enforce mathematical preconditions (such as positive-definiteness or symmetry) through C++20 concepts and runtime diagnostic evidence.

This started off as a personal compilation of my research and coursework code into a unified applied math library. I develop this package alongside downstream projects, continuously absorbing and refining new numerical tools into `numerics` for re-use. Because this has primarily been built for my own research workflows rather than by a large team, please use it with appropriate caution!

Over time, this package has grown to include everything from my undergraduate mesh-free fluid solvers (developed for surgical simulation) and master's work on graph algorithms and Ising nucleation, to my PhD research on finite state projection and iterative linear solvers.

Despite its organic evolution, the library is built on modern C++20 with 239 unit tests, clean fallback paths (from BLAS/LAPACK/OpenMP/CUDA acceleration down to pure portable C++).

Jump right in with @ref page_getting_started "Getting Started" or browse @ref page_examples "Examples".

---

## Core Interfaces

### 1. Direct Factorization and Solve
```cpp
#include <numerics.hpp>

num::Matrix A(2, 2, 0.0);
A(0, 0) = 4.0; A(0, 1) = 1.0;
A(1, 0) = 1.0; A(1, 1) = 3.0;

num::Vector b{1.0, 2.0};
num::Vector x(2, 0.0);

auto factor = num::cholesky(num::assume_spd(A));
num::cholesky_solve(factor, b, x); // Solves A * x = b
```

### 2. Matrix-Free Iterative Solvers
```cpp
#include <numerics.hpp>

// 5-point discrete Laplacian stencil on an N x N grid
auto laplacian = num::operators::make_op(
    [N](const num::Vector &u, num::Vector &Lu) {
        apply_fd_laplacian(u, Lu, N);
    }, N * N);

auto spd_L = num::operators::assume_spd(laplacian);
num::Vector u(N * N, 0.0);
num::cg(spd_L, rhs, u, 1e-8);
```

### 3. Unified Problem Dispatch
```cpp
#include <numerics.hpp>

auto op = num::operators::DenseOp(A);
auto solution = num::solve(
    num::LinearProblem{op, b},
    num::GMRES{.tol = 1e-10, .max_iter = 200});
```

---

## Architecture and Design Rules

1. **Deterministic Allocation:** Raw compute kernels operate on caller-provided output buffers; no hidden allocations in simulation loops.
2. **Layered Modules:** `kernel` has zero dependencies, `core` and `algebra` define types and concepts, and domain modules build on both (see @ref page_architecture).
3. **Storage / Operator Decoupling:** Solvers accept anything implementing the required mathematical protocol (`VectorSpace`, `LinearOperator`), whether stored as `Matrix` or evaluated on the fly via `make_op`.
4. **Hardware Acceleration:** Compiles against standard C++20 with optional runtime/compile-time dispatch for BLAS/LAPACK, OpenMP, SIMD, FFTW3, SuiteSparse, MPI, and CUDA.
5. **Enforced Invariants:** Algorithms state required properties (`SPDOperator`, `SelfAdjointOperator`). Passing an uncertified type fails at compile time; runtime claims (`assume_spd`) are validated under diagnostic presets (see @ref page_concepts).

---

## Documentation

1. @subpage page_getting_started "Getting Started" (CMake setup, header inclusion, basic operations)
2. @subpage page_architecture "Library Structure & Architecture" (standalone raw compute layer, module tiers, dependency invariants)
3. @subpage page_concepts "Concepts & Invariants" (algebraic concepts, property tags, diagnostic presets)
4. @subpage page_expressive "Expression Interface" (convenience operators and performance tradeoffs)
5. @subpage page_examples "Examples" (code organized by numerical domain)
6. @subpage page_reference "API Reference" (classes, functions, and concepts)
7. @subpage page_report "Benchmark Report" (kernel throughput, convergence, validation)
8. @subpage page_developer "Developer Documentation" (testing and contribution standards)

