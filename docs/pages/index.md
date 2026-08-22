# numerics {#mainpage}

Numerics is a C++20 scientific computing library providing dense and structured linear algebra, matrix-free operator calculus, differential equation integrators, spectral methods, and stochastic processes.

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
auto factor = num::cholesky(num::linalg::make_spd(A));

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
* **Storage and Operator Decoupling:** Concrete storage containers (`Vector`, `Matrix`, `SparseMatrix`) are decoupled from operator abstractions (`DenseOp`, `SparseOp`, `CallableOp`).
* **Multi-Backend Execution:** Algorithmic logic compiles against standard C++20 with optional acceleration paths for BLAS/LAPACK, OpenMP, SIMD/NEON, FFTW3, SuiteSparse, MPI, and CUDA.

---

## Concepts: Compile-Time Interface Contracts

C++20 Concepts enforce mathematical properties and interface contracts at compile time.

### Mathematical Contract Enforcement

Algorithms such as Conjugate Gradient require proof of symmetry and positive definiteness. Passing an unverified operator triggers a compile-time rejection:

```cpp
num::Matrix A(2, 2, 1.0);
num::operators::DenseOp op(A);

// --- 1. Violation: DenseOp does not declare the SPDLinearOperator contract ---
// num::cg(op, b, x); // Compile error!

// --- 2. Resolution: Verify properties and attach the required contract tag ---
auto spd_op = num::operators::assume_spd(op);
num::cg(spd_op, b, x); // Compiles and executes cleanly
```

**Compiler Diagnostic (Clang/GCC):**
```text
error: no matching function for call to 'cg'
note: candidate template ignored: constraints not satisfied [with Op = num::operators::DenseOp]
note: because 'SPDLinearOperator<num::operators::DenseOp>' evaluated to false
note: because 'SymmetricLinearOperator<num::operators::DenseOp>' evaluated to false:
      no member named 'symmetric_operator_tag' in 'num::operators::DenseOp'
```

### Concept Taxonomy

| Domain | Header | Concepts | Description |
| :--- | :--- | :--- | :--- |
| **Data Structures** | `core/concepts.hpp` | `Scalar`, `VectorLike`, `ContiguousVectorLike`, `DenseMatrixLike`, `SparseMatrixLike` | Memory layout and element access |
| **Operators** | `operator/concepts.hpp` | `LinearOperator`, `AdjointableLinearOperator`, `SPDLinearOperator`, `NonlinearOperator` | Matrix-free action and property tags |
| **Linear Algebra** | `linalg/concepts.hpp` | `Preconditioner`, `TriangularFactor`, `IsLinearSolver` | Iterative solvers and factorizations |
| **Differential Equations** | `ode/concepts.hpp` | `IsODEProblem`, `IsSymplecticODEProblem`, `VecField` | Initial-value and Hamiltonian systems |

---

## Diagnostics: Runtime Validation

`num::debug` provides runtime verification with exact source location attribution (`file:line:function`).

### 1. Mathematical Property Verification
When `DiagnosticLevel::full` is enabled, `assume_spd()` evaluates sampled quadratic forms $x^T A x$:

```cpp
num::Matrix A(2, 2, 0.0);
A(0, 0) = -5.0; // Indefinite matrix
A(1, 1) =  1.0;

num::operators::DenseOp op(A);
auto spd_op = num::operators::assume_spd(op); // Throws PropertyError at runtime
```

**Output:**
```text
[PropertyError] Error at include/operator/properties.hpp:67 in assume_spd:
  assume_spd() assertion failed: sampled inner product x^T A x = -5.000000 <= 0.
  The operator is NOT positive definite!
```

### 2. Dimension and Data Invariants
```cpp
// Dimension mismatch
num::debug::check_dim(expected_dim, actual_dim, "state vector");
// Output: [DimensionError] Error at src/solvers/cg.cpp:21: expected 100, got 50

// Non-finite value check
num::debug::check_finite(buffer, size, "solution array");
// Output: [ValueError] Error at include/core/debug.hpp:75: non-finite value at index 3

// Sparse structure validation (CSR monotonicity and index bounds)
num::debug::verify_sparse_structure(sparse_matrix);
// Output: [SparseStructureError] Error at include/core/debug.hpp:160: col_idx exceeds n_cols
```

---

## Documentation

- @subpage page_getting_started "Getting Started" — compilation, headers, and basic operations.
- @subpage page_examples "Feature Examples" — code examples organized by numerical domain.
- @subpage page_guides "Guides" — solver selection, performance profiling, and workflows.
- @subpage page_reference "API Reference" — detailed class, function, and concept documentation.
- @subpage page_developer "Developer Documentation" — architecture, testing, and contribution standards.
- [GitHub Repository](https://github.com/AdityaDendukuri/numerics) — source code, issues, and releases.
