# numerics

Numerics is a C++20 scientific computing library providing dense and structured linear algebra, matrix-free operator calculus, differential equation integrators, spectral methods, and stochastic processes. It pairs compile-time concept enforcement with runtime contract validation.

- **Online Documentation:** [adityadendukuri.github.io/numerics](https://adityadendukuri.github.io/numerics/)
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

## CMake Integration

### 1. Using via `FetchContent` (Recommended)

Add `numerics` directly to your `CMakeLists.txt`:

```cmake
include(FetchContent)
FetchContent_Declare(
    numerics
    GIT_REPOSITORY https://github.com/AdityaDendukuri/numerics.git
    GIT_TAG        main
)
FetchContent_MakeAvailable(numerics)

# Link only the modular targets your application requires
target_link_libraries(my_application PRIVATE numerics::solvers numerics::plot)
```

### 2. System Installation & `find_package`

Build and install `numerics`:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local
cmake --build build -j
sudo cmake --install build
```

Then in your downstream `CMakeLists.txt`:

```cmake
find_package(numerics REQUIRED)
target_link_libraries(my_application PRIVATE numerics::numerics)
```

---

## Modular CMake Targets

Numerics is partitioned into focused, granular targets so consumers only pay the compilation and linking cost of the components they use:

| Target | Description | Dependencies |
| :--- | :--- | :--- |
| **`numerics::numerics`** | Monolithic umbrella library containing all modules | All enabled backends |
| **`numerics::kernel`** | Core vectors, dense/sparse/banded matrices, and linear operator interfaces | None (Standard C++20) |
| **`numerics::solvers`** | Factorizations (LU, QR, Cholesky, Thomas), Krylov solvers (CG, GMRES, PCG, MINRES), Eigen/SVD, Matrix exponentials | `numerics::kernel`, BLAS/LAPACK (optional) |
| **`numerics::ode`** | Time integrators: Euler, RK4, RK45, Verlet, Yoshida4, and implicit integrators | `numerics::kernel` |
| **`numerics::pde`** | Structured grids, finite-difference stencils, Poisson, and diffusion field solvers | `numerics::kernel`, `numerics::solvers` |
| **`numerics::spectral`** | Complex and real multidimensional FFT transforms | FFTW3 (optional), Accelerate/SIMD |
| **`numerics::plot`** | Header-only publication figure export (Gnuplot) and terminal ASCII plotting | None |
| **`numerics::io`** | JSON parser and sparse matrix disk serialization (`nlohmann/json`) | `numerics::kernel` |
| **`numerics::raw_kernel`** | Header-only Level 1 raw array kernels | None |

---

## CMake Configuration Options

The build configuration can be customized via standard CMake flags:

| Option | Default | Description |
| :--- | :---: | :--- |
| `NUMERICS_BUILD_TESTS` | `OFF` | Build the unit test suite (`numerics_tests`) |
| `NUMERICS_BUILD_EXAMPLES` | `ON` (top-level) | Build example executables |
| `NUMERICS_BUILD_DOCS` | `OFF` | Build Doxygen documentation website |
| `NUMERICS_USE_BLAS` | `ON` | Enable optimized BLAS backend (`Accelerate` / `OpenBLAS`) |
| `NUMERICS_USE_LAPACK` | `ON` | Enable LAPACK factorization routines (`LAPACKE`) |
| `NUMERICS_USE_OPENMP` | `ON` | Enable multi-threaded OpenMP acceleration |
| `NUMERICS_USE_FFTW` | `ON` | Link FFTW3 for high-performance spectral transforms |
| `NUMERICS_USE_SUITESPARSE` | `ON` | Enable SuiteSparse KLU and UMFPACK direct sparse solvers |
| `NUMERICS_BUILD_IO` | `ON` | Build the optional JSON file-I/O module |
| `NUMERICS_ENABLE_MPI` | `ON` | Enable distributed MPI operations |
| `NUMERICS_ENABLE_CUDA` | `ON` | Enable CUDA GPU acceleration |

---

## Documentation

Full interactive documentation, tutorials, and API reference are hosted online:
* **Documentation Website:** [https://adityadendukuri.github.io/numerics/](https://adityadendukuri.github.io/numerics/)
* **Build Docs Locally:** Run `cmake --build build --target docs` with Doxygen installed.

---

## License

Numerics is released under the MIT License.
