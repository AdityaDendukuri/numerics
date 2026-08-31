# Getting Started {#page_getting_started}

This guide introduces the core design philosophy of Numerics, its primary data structures, execution styles, mathematical invariant framework, and linear solvers.

---

## 1. Add Numerics to Your Project

Numerics is a modern C++20 header-first numerical computing library.

### CMake Integration
```cmake
find_package(numerics REQUIRED)
target_link_libraries(my_program PRIVATE numerics::numerics)
```

For pure dependency-free deployments containing only containers, algebraic concepts, and fallback numerical kernels, link against `numerics::core`:
```cmake
find_package(numerics REQUIRED COMPONENTS core)
target_link_libraries(my_program PRIVATE numerics::core)
```

Optional hardware acceleration backends are available as modular targets: `numerics::blas`, `numerics::lapack`, `numerics::openmp`, `numerics::fftw`, `numerics::suitesparse`, `numerics::mpi`, and `numerics::cuda`.

### Header Inclusion
Include the umbrella header to access the complete public API:
```cpp
#include <numerics.hpp>
```

---

## 2. Core Data Structures

Numerics provides cache-aligned, continuous memory containers for linear algebra and numerical routines:

```cpp
#include <numerics.hpp>

// Standard direct construction
num::Vector x{1.0, 2.0, 3.0}; // Length-3 vector
num::Matrix A(3, 3, 0.0);      // 3x3 row-major dense matrix initialized to zero

// Matrix and vector element access
A(0, 0) = 4.0;
A(0, 1) = 1.0;
x[0] = 2.0;

// Factory constructors and utilities
num::Matrix Z = num::zeros(3, 3);       // Zero matrix
num::Matrix I = num::eye(3);            // Identity matrix
num::Vector v = num::linspace(0.0, 1.0, 5); // [0.0, 0.25, 0.5, 0.75, 1.0]
num::real s   = num::accu(A);           // Sum of all elements
```

---

## 3. Two Operating Styles: Expressions vs. Zero-Allocation Kernels

Depending on your workflow, Numerics offers two complementary execution models:

### Rapid Mathematical Prototyping (num::ops)
For rapid prototyping, test assertions, and textbook formula readability, enable the `num::ops` namespace to evaluate natural value-returning infix expressions:

```cpp
using namespace num::ops;

num::Matrix A = num::ones(3, 3);
num::Matrix B = num::eye(3);
num::Vector x{1.0, 2.0, 3.0};

// Natural algebraic expressions
num::Matrix C = A * B + 2.0 * B;
num::Vector y = A * x - x / 2.0;
```

### High-Performance Zero-Allocation Kernels (Production Simulations)
In performance-critical simulation loops, ODE integrators, and inner iterative solvers executing millions of steps, dynamic heap allocations on every binary operator create allocator contention and memory bandwidth bottlenecks. The production idiom pre-allocates destination buffers once and uses mutating out-parameter kernels:

```cpp
// Allocate once outside the simulation loop
num::Vector y(3, 0.0);
num::Vector z(3, 1.0);

for (num::idx step = 0; step < total_steps; ++step) {
    // Zero dynamic allocations inside the loop
    num::matvec(A, x, y); // y = A * x
    num::axpy(2.0, z, y); // y = y + 2.0 * z
}
```

See @ref page_expressive "Expression Interface" for an in-depth discussion on performance tradeoffs.

---

## 4. Mathematical Invariants & Evidence-Based Solvers

In numerical computing, specialized algorithms mathematically require specific operator properties to guarantee convergence and stability. For example:
* **Conjugate Gradient (`num::cg`)** mathematically requires the system to be **Symmetric Positive Definite (SPD)**: \f$A = A^T\f$ and \f$x^T A x > 0\f$.
* **Cholesky Factorization (`num::cholesky`)** requires SPD matrices to guarantee real, positive diagonal pivots.
* **MINRES (`num::minres`)** requires **Symmetric / Self-Adjoint** operators (\f$A = A^T\f$).

Passing an uncertified general matrix to `num::cg` or `num::cholesky` produces a **compile-time concept failure**, preventing catastrophic runtime divergence.

### Attaching Invariant Evidence to General Matrices

When you know from domain physics that a matrix is positive-definite, attach evidence explicitly:

```cpp
num::Matrix A(3, 3, 0.0);
// fill symmetric positive-definite entries...

// 1. Tag by claim (verified probabilistically under active diagnostic preset)
auto spd_A = num::assume_spd(A);

// 2. Or tag by exhaustive O(n^3) validation (throws if not SPD)
auto spd_validated = num::make_spd(A);

// Now accepted by CG and Cholesky
num::Vector b{1.0, 2.0, 3.0};
num::Vector x(3, 0.0);
num::cg(spd_A, b, x);
```

### Operators That Carry Invariants by Construction

Physical discretizations that mathematically guarantee a property carry proof in their type automatically:

```cpp
const num::Grid2D grid{32, 1.0 / 33.0};

// A backward-Euler discretization of Dirichlet diffusion is SPD by construction:
const num::operators::BackwardEuler2D system(grid.N, /*dt=*/0.05);

num::Vector rhs(grid.size(), 1.0);
num::Vector solution(grid.size(), 0.0);

// Accepted directly by CG without any manual assume_spd() tagging:
const auto result = num::cg(system, rhs, solution);
```

See @ref page_concepts "Concepts, Invariants & Diagnostics" for details on property lattices and diagnostic presets.

---

## 5. Direct Factorizations & Iterative Solvers

### Direct Solvers (Factorize Once, Solve Many)
```cpp
// Cholesky factorization for SPD systems
auto factor = num::cholesky(num::assume_spd(A));
num::cholesky_solve(factor, b, x); // Solves A * x = b in O(n^2)

// LU factorization with partial pivoting for general square systems
auto lu_factor = num::lu(num::assume_square(A));
num::lu_solve(lu_factor, b, x);
```

### High-Level Problem Dispatch (num::solve)
For high-level algorithms and configuration-driven workflows, Numerics provides a unified problem abstraction:

```cpp
auto op = num::operators::DenseOp(A);
auto result = num::solve(
    num::LinearProblem{op, b},
    num::GMRES{.tol = 1e-10, .max_iter = 200});
```

---

## 6. Standalone Raw Compute Tier (num::kernel::raw)

If your project already manages its own memory (via raw pointers `double*`, `std::vector`, Eigen matrices, or custom buffers), operates under real-time / embedded constraints, or requires zero dynamic heap allocations and zero external dependencies, you can directly use the standalone Tier-0 compute layer:

```cpp
#include <kernel/factor.hpp>
#include <kernel/krylov.hpp>

// Solve A * x = b directly over caller-owned pointers:
std::vector<double> A = {4.0, 1.0, 1.0, 3.0};
std::vector<double> L(4, 0.0), b = {1.0, 2.0}, x(2, 0.0);

if (num::kernel::raw::cholesky(L.data(), A.data(), 2)) {
    num::kernel::raw::cholesky_solve(x.data(), L.data(), b.data(), 2);
}
```

See @ref page_architecture "Library Structure & Architecture" for details on the tiered hierarchy, how to vendor `include/kernel/`, and CMake integration.

---

## 7. Result Introspection & Terminal Documentation

### In-Code Stream Printing
All solver and algorithm result structures (`num::SolverResult`, `num::kernel::raw::KrylovResult`, `num::ODEResult`, `num::SymplecticResult`, `num::RootResult`, `num::SVDResult`, `num::EigenResult`, `num::PowerResult`, `num::BandedSolverResult`, `num::ClusterResult`) implement standard `operator<<` stream formatting:

```cpp
auto res = num::cg(A, b, x);
std::cout << res << "\n";
// Output: SolverResult{ converged: true, iterations: 24, residual: 1.42e-11 }
```

### CLI Documentation Lookup
Building the documentation target automatically compiles Section-3 UNIX man pages into `build/docs/man/`:

```bash
# 1. Build documentation and man pages
cmake --build build --target docs

# 2. Query any symbol with the included helper script:
./tools/doc SolverResult
./tools/doc KrylovResult
./tools/doc ODEResult
./tools/doc cg

# Or query directly via standard man:
man build/docs/man/man3/num_SolverResult.3
```

---

## 8. Next Steps & Detailed Guides

Explore dedicated guides for each numerical domain:

* @ref page_architecture "Library Structure & Architecture" — Standalone raw kernel layer, tiered hierarchy, and dependency invariants.
* @ref page_concepts "Mathematical Concepts & Diagnostics Framework" — Type-level laws, invariant tags, and diagnostic presets.
* @ref page_linear "Linear Algebra Guide" — Direct factorizations, Krylov methods, eigenvalue/SVD algorithms, and banded solvers.
* @ref page_solver_best_practices "Linear Solver Selection Guide" — Decision trees for picking optimal direct vs. iterative methods.
* @ref page_ode "Ordinary Differential Equations" — Explicit, adaptive RK45, and symplectic Verlet/Yoshida integrators.
* @ref page_quadrature "Quadrature & Integration" — Composite, Gaussian, adaptive, and Talbot contour integration.
* @ref page_spectral "Spectral Methods & FFT" — Fast Fourier transforms, Poisson solvers, and discrete sine transforms.
* @ref page_stochastic "Stochastic Methods" — Metropolis–Hastings sampling, Boltzmann tables, and umbrella sampling.
* @ref page_expressive "Expression Interface" — Infix operators and zero-allocation idioms.
* @ref page_examples "Browse Runnable Examples" — Code organized by numerical domain.
* @ref page_reference "API Reference" — Generated index of classes, functions, and concepts.
* @ref page_report "Performance Benchmark Report" — Empirical benchmarks and comparisons against vendor LAPACK.

