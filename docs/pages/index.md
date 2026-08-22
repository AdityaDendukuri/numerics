# numerics {#mainpage}

Numerics is a modular C++20 library for dense and structured linear algebra,
matrix-free solvers, ODE/PDE integration, spectral transforms, and stochastic
methods.

## First Solve

```cpp
#include <numerics.hpp>

num::Matrix A(2, 2, 0.0); // Allocate a 2-by-2 matrix.
A(0, 0) = 4.0;
A(0, 1) = 1.0;
A(1, 0) = 1.0;
A(1, 1) = 3.0;

num::Vector b{1.0, 2.0}; // Define the right-hand side.
auto factor = num::cholesky(num::linalg::make_spd(A));

num::Vector x;
num::cholesky_solve(factor, b, x); // Solve A*x=b.
```

## Library Overview

| Area | Included functionality |
| --- | --- |
| Core | Owning vectors and matrices, views, reductions, indexing, and execution backends |
| Linear algebra | Dense and banded factorizations, sparse-direct methods, Krylov solvers, eigenproblems, SVD, resolvents, and matrix exponentials |
| Differential equations | Explicit, adaptive, symplectic, and implicit integration |
| PDEs and fields | Structured grids, finite-difference stencils, diffusion, Poisson solves, and vector calculus |
| Spectral methods | Complex and real Fourier transforms with reusable plans |
| Stochastic methods | Categorical sampling, Metropolis and umbrella sweeps, histograms, and online statistics |
| Spatial utilities | Connected components, periodic lattices, neighbor lists, and SPH kernels |

## Working with Numerics

Containers own contiguous storage. Views and operators borrow existing objects.
Low-level routines write into caller-provided outputs so allocations can be
reused, while factorizations and high-level solves return result objects with
convergence metadata.

```cpp
num::Vector y(A.n_rows());
num::matvec(A, x, y); // Reuse y across repeated products.

auto operator_A = num::operators::DenseOp(A); // Borrow A without copying it.
auto solution = num::solve(
    num::LinearProblem{operator_A, b},
    num::GMRES{.tol = 1e-10}); // Return the solution and convergence data.
```

Sequential implementations are always available. OpenMP, SIMD, BLAS/LAPACK,
FFTW, CUDA, MPI, KLU, and UMFPACK paths are selected when their dependencies are
enabled and available.

## Concepts and Diagnostics

Concepts check interfaces and declared properties during compilation. Diagnostics
check dimensions, values, and sampled operator properties while the program runs.

```cpp
num::operators::DenseOp op(A);
static_assert(num::LinearOperator<decltype(op)>); // apply(), rows(), and cols() exist.

num::debug::set_level(num::debug::DiagnosticLevel::full);
auto spd_op = num::operators::assume_spd(op); // Sample x^T*A*x before adding the SPD tag.

static_assert(num::SPDLinearOperator<decltype(spd_op)>); // CG now accepts the operator.
num::SolverResult result = num::cg(spd_op, b, x);
```

The concept prevents an incompatible call from compiling. The diagnostic catches
invalid runtime data before the tagged operator reaches the solver. Sampled checks
are useful guards, not mathematical proofs.

## Documentation

- @subpage page_getting_started "Getting Started" — linking, headers, objects, and the first operation.
- @subpage page_examples "Feature Examples" — focused snippets covering individual library features.
- @subpage page_guides "Guides" — solver selection, performance, and complete numerical workflows.
- @subpage page_reference "API Reference" — namespaces, types, functions, concepts, and source files.
- @subpage page_developer "Developer Documentation" — repository structure, validation, and contribution workflow.
