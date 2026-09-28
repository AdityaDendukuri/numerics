# Getting Started

This page covers installation, the containers, the two programming styles, the laws solvers require, and the direct and iterative solvers.

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

Vectors and matrices are dense containers with contiguous storage.
Every routine in the library accepts them or a view of them.

```cpp
#include <numerics.hpp>

// Standard direct construction
num::vec x{1.0, 2.0, 3.0}; // Length-3 vector
num::mat A(3, 3, 0.0);      // 3x3 row-major dense matrix initialized to zero

// mat and vector element access
A(0, 0) = 4.0;
A(0, 1) = 1.0;
x[0] = 2.0;

// Factory constructors and utilities
num::mat Z = num::zeros(3, 3);       // Zero matrix
num::mat I = num::eye(3);            // Identity matrix
num::vec v = num::linspace(0.0, 1.0, 5); // [0.0, 0.25, 0.5, 0.75, 1.0]
num::real s   = num::accu(A);           // Sum of all elements
```

---

## 3. Two Operating Styles: Expressions vs. Zero-Allocation Kernels

Depending on your workflow, Numerics offers two complementary execution models:

### Rapid Mathematical Prototyping (num::ops)
For rapid prototyping, test assertions, and textbook formula readability, enable the `num::ops` namespace to evaluate natural value-returning infix expressions:

```cpp
using namespace num::ops;

num::mat A = num::ones(3, 3);
num::mat B = num::eye(3);
num::vec x{1.0, 2.0, 3.0};

// Natural algebraic expressions
num::mat C = A * B + 2.0 * B;
num::vec y = A * x - x / 2.0;
```

### High-Performance Zero-Allocation Kernels (Production Simulations)
In performance-critical simulation loops, ODE integrators, and inner iterative solvers executing millions of steps, dynamic heap allocations on every binary operator create allocator contention and memory bandwidth bottlenecks. The production idiom pre-allocates destination buffers once and uses mutating out-parameter kernels:

```cpp
// Allocate once outside the simulation loop
num::vec y(3, 0.0);
num::vec z(3, 1.0);

for (num::idx step = 0; step < total_steps; ++step) {
    // Zero dynamic allocations inside the loop
    num::matvec(A, x, y); // y = A * x
    num::axpy(2.0, z, y); // y = y + 2.0 * z
}
```

---

## 4. Laws and Solvers

In numerical computing, specialized algorithms mathematically require specific operator properties to guarantee convergence and stability. For example:
* **Conjugate Gradient (`num::cg`)** mathematically requires the system to be **Symmetric Positive Definite (SPD)**: $A = A^T$ and $x^T A x > 0$.
* **Cholesky Factorization (`num::cholesky`)** requires SPD matrices to guarantee real, positive diagonal pivots.
* **MINRES (`num::minres`)** requires **Symmetric / Self-Adjoint** operators ($A = A^T$).

Passing a matrix without the required law to `num::cg` or `num::cholesky` produces a **compile-time concept failure**, preventing catastrophic runtime divergence.

### Attaching a Law to a Matrix

When you know from domain physics that a matrix is positive-definite, attach the law explicitly:

```cpp
num::mat A(3, 3, 0.0);
// fill symmetric positive-definite entries...

// 1. Tag by claim (verified probabilistically under active diagnostic preset)
auto spd_A = num::assume_spd(A);

// 2. Or tag by exhaustive O(n^3) validation (throws if not SPD)
auto spd_validated = num::make_spd(A);

// Now accepted by CG and Cholesky
num::vec b{1.0, 2.0, 3.0};
num::vec x(3, 0.0);
num::cg(spd_A, b, x);
```

### Operators That Carry Invariants by Construction

Physical discretizations that mathematically guarantee a property carry proof in their type automatically:

```cpp
const num::grid2d grid{32, 1.0 / 33.0};

// A backward-Euler discretization of Dirichlet diffusion is SPD by construction:
const num::operators::backward_euler_2d system(grid.N, /*dt=*/0.05);

num::vec rhs(grid.size(), 1.0);
num::vec solution(grid.size(), 0.0);

// Accepted directly by CG without any manual assume_spd() tagging:
const auto result = num::cg(system, rhs, solution);
```

See [Concepts, Laws & Diagnostics](concepts.md) for the laws and the diagnostic presets.

---

## 5. Direct Factorizations & Iterative Solvers

### Direct Solvers (Factorize Once, Solve Many)
```cpp
// Cholesky factorization for SPD systems
auto factor = num::cholesky(num::assume_spd(A));
num::cholesky_solve(factor, b, x); // Solves A * x = b in O(n^2)

// LU factorization with partial pivoting for general square systems
auto lu_factor = num::lu(A);
num::lu_solve(lu_factor, b, x);
```

---

## 6. Standalone Raw Compute Tier (num::kernel)

If your project already manages its own memory (via raw pointers `double*`, `std::vector`, Eigen matrices, or custom buffers), operates under real-time / embedded constraints, or requires zero dynamic heap allocations and zero external dependencies, you can directly use the standalone Tier-0 compute layer:

```cpp
#include <kernel/factor.hpp>
#include <kernel/krylov.hpp>

// Solve A * x = b directly over caller-owned pointers:
std::vector<double> A = {4.0, 1.0, 1.0, 3.0};
std::vector<double> L(4, 0.0), b = {1.0, 2.0}, x(2, 0.0);

if (num::kernel::cholesky(L.data(), A.data(), 2)) {
    num::kernel::cholesky_solve(x.data(), L.data(), b.data(), 2);
}
```

See [Library Structure & Architecture](architecture.md) for details on the tiered hierarchy, how to vendor `include/kernel/`, and CMake integration.

---

## 7. Printing Results

Every solver and algorithm result (`num::solver_result`, `num::kernel::krylov_result`, `num::ode_result`, `num::symplectic_result`, `num::root_result`, `num::svd_result`, `num::eigen_result`, `num::power_result`, `num::banded_solver_result`, `num::cluster_result`) has an `operator<<`:

```cpp
auto res = num::cg(A, b, x);
std::cout << res << "\n";
// solver_result{ converged: true, iterations: 24, residual: 1.42e-11 }
```

---

## 8. Next Steps

* [Concepts, Laws & Diagnostics](concepts.md): what each algorithm requires, and how a type states it.
* [Architecture](architecture.md): the module tiers, and how to vendor `include/kernel/`.
* [Reference](reference/index.md): one page per public name, grouped by topic and by directory.
* [Examples](reference/examples/index.md): complete programs.
