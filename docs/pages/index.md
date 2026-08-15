# numerics {#mainpage}

Modular C++20 numerical kernel and solver suite for dense and structured linear algebra, matrix-free Krylov methods, ODE/PDE integrators, and spectral transforms.

---

## Table of Contents

- [Three Layers & Target Dependencies](#three-layers--target-dependencies)
- [Code Examples by Layer](#code-examples-by-layer)
- [C++20 Concept Enforcement](#c20-concept-enforcement)
- [Runtime Diagnostics](#runtime-diagnostics)
- [Documentation Index](#documentation-index)
  - [Core Module Guides](#core-module-guides)
  - [Spatial & Domain Applications](#spatial--domain-applications)
  - [Implementation & Developer Guides](#implementation--developer-guides)

---

## Three Layers & Target Dependencies

```text
                     numerics::kernel  (Layer 1 & 2: Vectors, Matrices, Fields, Operators)
                      /      |      \
                     /       |       \
                    v        |        v
   numerics::spectral        |       numerics::solvers (LU, QR, Cholesky, CG, GMRES, SVD)
                             |        |
                             |        v
                             +---> numerics::ode (RK4, RK45, Verlet, Yoshida4)
                                      |
                                      v
                                  numerics::pde (FieldSolver, Poisson DST-I)
```

| Layer | CMake Target | Components | Recommended Use |
| :--- | :--- | :--- | :--- |
| **Layer 1** | `numerics::raw_kernel` | Header-only raw loops and memory routines | Use for zero-overhead inline memory loops without library compilation. |
| **Layer 2** | `numerics::kernel` | Data structures & operators (`Vector`, `Matrix`, `SparseMatrix`, `BandedMatrix`, `Fields`, `LinearOperator`, `assume_spd()`) | Use when application requires arrays, grids, and operator abstractions without solver overhead or external dependencies. |
| **Layer 3** | `numerics::numerics` | Full solver suite (`solve()`, LU/QR/SVD, CG, GMRES, RK45, PDE, FFT) | Use when complete linear, differential, or spectral solvers are required. |

---

## Code Examples by Layer

### 1. Storage & Core Data Structures (Layer 1 & 2)
```cpp
#include <numerics.hpp>

num::Vector x{1.0, 2.0, 3.0};
num::Matrix A(3, 3, 0.0);
A(0, 0) = 4.0; A(0, 1) = 1.0;
A(1, 0) = 1.0; A(1, 1) = 4.0; A(1, 2) = 1.0;
A(2, 1) = 1.0; A(2, 2) = 4.0;
```

### 2. Operators & Property Tags (Layer 2)
```cpp
#include <numerics.hpp>

num::operators::DenseOp Aop(A);
auto spd_A = num::operators::assume_spd(Aop);
```

### 3. Solvers & Numerical Integrators (Layer 3)
```cpp
#include <numerics.hpp>

// Direct LU Factorization
auto fact = num::lu(A);
num::Vector sol;
num::lu_solve(fact, b, sol);

// Iterative Krylov Solver
num::LinearSolution s = num::solve(num::LinearProblem{spd_A, b}, num::CG{});

// Adaptive ODE Integration (RK45)
auto rhs = [](double t, const num::Vector& y, num::Vector& dy) {
    dy[0] = y[1];
    dy[1] = -y[0];
};
auto ode_res = num::solve(num::ODEProblem{rhs, {1.0, 0.0}, 0.0, 10.0}, num::RK45{});
```

---

## C++20 Concept Enforcement

```cpp
static_assert(num::LinearOperator<decltype(Aop)>);

// CG requires an SPD operator concept guard
auto spd_A = num::operators::assume_spd(Aop);
static_assert(num::SPDLinearOperator<decltype(spd_A)>);

num::LinearSolution s = num::solve(num::LinearProblem{spd_A, b}, num::CG{});
```

---

## Runtime Diagnostics

```cpp
num::debug::set_level(num::debug::DiagnosticLevel::full);

// Catches dimension mismatches, non-finite values, and invalid mathematical assertions:
// [PropertyError] Error at main.cpp:14 in main:
//   assume_spd() assertion failed: sampled inner product x^T A x = -4.000000 <= 0.
```

---

## Documentation Index

### Core Module Guides
- @subpage page_linalg "Linear Algebra & Factorizations (LU, QR, Cholesky, SVD, Eigen, Arnoldi/expv)"
- @subpage page_operators "Linear Operators & Matrix-Free Krylov Solvers (CG, PCG, MINRES, GMRES)"
- @subpage page_solver_best_practices "Solver Selection Taxonomy & Best Practices Guide"
- @subpage page_fft "Spectral Transforms & Reusable FFT Plans (Cooley-Tukey, SIMD, FFTW3)"
- @subpage page_ode "ODE Steppers & Integrators (RK4, Adaptive RK45, Symplectic Verlet/Yoshida4)"
- @subpage page_pde "PDE Discretizations & ADI Diffusion Operators"
- @subpage page_poisson "Dirichlet Poisson Solve via Discrete Sine Transform (DST-I)"
- @subpage page_analysis "Numerical Quadrature & Root Finding"
- @subpage page_stochastic "Stochastic Methods, MCMC & Statistical Analysis"

### Spatial & Domain Applications
- @subpage page_fields "3D Spatial Fields & Vector Differential Operators"
- @subpage page_stencil_hof "Higher-Order Stencil Operators"
- @subpage page_sph_kernel "Smoothed Particle Hydrodynamics (SPH) Density & Pressure Kernels"
- @subpage page_pbc_lattice "Periodic Square & Cubic Lattice Indexing"
- @subpage page_connected_components "Grid Connected Component Labeling"

### Implementation & Developer Guides
- @subpage page_parallel "Parallel Backend Boundaries (OpenMP, CUDA, MPI)"
- @subpage page_performance "Benchmarking & Performance Methodology"
- @subpage page_ns_perf "Navier-Stokes Projection Solver Pattern"
- @subpage page_developer_workflow "Developer Setup, Tag Generation & Local API Search"
