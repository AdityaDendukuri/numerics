# Library Structure & Architecture {#page_architecture}

Numerics is organized into strict, unidirectional architectural tiers. Each tier depends only on the tiers below it, enabling any layer—especially the standalone raw compute kernel tier—to be directly included, tested, or vendored into foreign codebases without pulling in unnecessary dependencies.

---

## 1. Architectural Tiers

```
┌─────────────────────────────────────────────────────────────┐
│ Tier 4: Problem Dispatch (num::solve, LinearProblem)        │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│ Tier 3: Domain Modules (linear, ode, pde, spectral, etc.)   │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│ Tier 2: Containers & Operators (Vector, Matrix, DenseOp)    │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│ Tier 1: Concepts & Invariants (algebra, core, diagnostics)  │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│ Tier 0: Standalone Raw Compute Kernels (num::kernel::raw)   │
└─────────────────────────────────────────────────────────────┘
```

| Tier | Module | Responsibilities | Dependencies |
| :--- | :--- | :--- | :--- |
| **Tier 0** | `kernel` | Raw compute over pointers (`T*`), dimensions, and callables; zero allocations | *None* (Pure Standard C++20) |
| **Tier 1** | `core`, `algebra` | Scalar fields, vector spaces, property lattice, and runtime diagnostic evidence | `kernel` |
| **Tier 2** | `container`, `operator` | `Vector`, `Matrix`, `SparseMatrix`, matrix-free operators, hardware backend dispatch | `core`, `algebra`, `kernel` |
| **Tier 3** | `linear`, `ode`, `pde`, `spectral`, `quadrature`, `roots`, `structures`, `spatial`, `stochastic` | Numerical domain algorithms (factorizations, Krylov solvers, RK45, Verlet, FFT, graph structures) | Tier 0–2 |
| **Tier 4** | `solve` | Unified problem dispatch (`LinearProblem`, `ODEProblem`) | All tiers |
| **Auxiliary** | `io`, `plot`, `viz` | Optional I/O and terminal plotting utilities | Independent |

---

## 2. The Standalone Raw Compute Layer (num::kernel::raw)

The bottom tier (`include/kernel/`) is a completely self-contained mathematical compute engine. It operates exclusively on raw pointers (`T*`), leading dimensions, caller-provided scratch workspaces, and generic callables.

### Key Characteristics of Tier 0
* **Zero External Dependencies:** Depends strictly on standard C++ library headers (`<algorithm>`, `<cmath>`, `<complex>`, `<concepts>`, `<cstddef>`).
* **Zero Dynamic Heap Allocations:** Never calls `new`, `malloc`, or allocates heap buffers. All temporary workspace is caller-managed (`T* work`).
* **Foreign Type Agnostic:** Works seamlessly with `std::vector<T>`, `std::array<T, N>`, Eigen vectors (`v.data()`), Armadillo matrices (`M.memptr()`), PyTorch/CUDA host tensors, or raw heap/stack buffers.
* **100% Copyable / Vendorable:** Drop `include/kernel/` directly into any embedded, real-time, game engine, or legacy codebase.

### Including and Linking Tier 0

Include the umbrella raw kernel header:
```cpp
#include <kernel/kernel.hpp>
```

Or individual modular kernel headers:
* `<kernel/raw.hpp>`: BLAS-1/2/3 vector and matrix kernels, Givens plane rotations, triangular solves (`trsv`, `trsm`), and Gram–Schmidt orthogonalization.
* `<kernel/factor.hpp>`: Cholesky ($A = LL^T$), blocked Cholesky, LU with partial pivoting ($PA = LU$), blocked LU, shifted Hessenberg factorizations, and banded LU solvers.
* `<kernel/krylov.hpp>`: Matrix-free Conjugate Gradient (`cg`) and Preconditioned Conjugate Gradient (`pcg`) operating on generic callable operators.

In CMake:
```cmake
find_package(numerics REQUIRED COMPONENTS kernel)
target_link_libraries(my_program PRIVATE numerics::kernel)
```

---

## 3. Standalone Raw Kernel Usage Examples

### Example 1: Raw-Pointer BLAS and Matrix-Vector Multiplication
```cpp
#include <iostream>
#include <vector>
#include <kernel/raw.hpp>

int main() {
    constexpr num::idx n = 4;
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> y = {0.5, 1.5, 2.5, 3.5};
    std::vector<double> z(n, 0.0);

    // 1. Vector linear combination: z = 2.0 * x + 3.0 * y
    num::kernel::raw::axpbyz(z.data(), x.data(), y.data(), 2.0, 3.0, n);

    // 2. Dot product: s = x . y
    double dot_val = num::kernel::raw::dot(x.data(), y.data(), n);

    // 3. Euclidean norm: ||x||_2
    double norm_val = num::kernel::raw::norm(x.data(), n);

    // 4. Dense matrix-vector product: y = A * x (A is 4x4 row-major)
    std::vector<double> A = {
        4.0, 1.0, 0.0, 0.0,
        1.0, 4.0, 1.0, 0.0,
        0.0, 1.0, 4.0, 1.0,
        0.0, 0.0, 1.0, 4.0
    };
    std::vector<double> Ax(n, 0.0);
    num::kernel::raw::matvec(Ax.data(), A.data(), x.data(), /*m=*/n, /*n=*/n);

    std::cout << "Dot: " << dot_val << ", Norm: " << norm_val << ", Ax[0]: " << Ax[0] << "\n";
}
```

### Example 2: In-Place Cholesky and LU Factorization Over Raw Buffers
```cpp
#include <iostream>
#include <vector>
#include <kernel/factor.hpp>

int main() {
    constexpr num::idx n = 3;

    // Symmetric Positive Definite 3x3 matrix (row-major)
    std::vector<double> A = {
        4.0, 2.0, -1.0,
        2.0, 5.0,  1.0,
       -1.0, 1.0,  6.0
    };
    std::vector<double> L(n * n, 0.0);
    std::vector<double> b = {1.0, 2.0, 3.0};
    std::vector<double> x(n, 0.0);

    // 1. Cholesky factorization: A = L * L^T
    bool spd = num::kernel::raw::cholesky(L.data(), A.data(), n);
    if (spd) {
        // Solves A * x = b via forward/back substitution through L
        num::kernel::raw::cholesky_solve(x.data(), L.data(), b.data(), n);
        std::cout << "Cholesky Solution: [" << x[0] << ", " << x[1] << ", " << x[2] << "]\n";
    }

    // 2. General LU Factorization with partial pivoting: P * A = L * U
    std::vector<double> LU = A; // Factorization is computed in-place
    std::vector<num::idx> ipiv(n, 0);
    bool nonsingular = num::kernel::raw::lu_factor(LU.data(), ipiv.data(), n);
    if (nonsingular) {
        num::kernel::raw::lu_solve(x.data(), LU.data(), ipiv.data(), b.data(), n);
        std::cout << "LU Solution: [" << x[0] << ", " << x[1] << ", " << x[2] << "]\n";
    }
}
```

### Example 3: Standalone Matrix-Free Conjugate Gradient
```cpp
#include <iostream>
#include <vector>
#include <kernel/krylov.hpp>

int main() {
    constexpr num::idx n = 1000;
    std::vector<double> b(n, 1.0);
    std::vector<double> x(n, 0.0);

    // Caller allocates required scratch workspace (3 * n elements for standard CG)
    std::vector<double> work(3 * n, 0.0);

    // Matrix-free 1D discrete Laplacian operator: y = -u''(x)
    auto apply_laplacian = [n](const double *u, double *Lu) {
        for (num::idx i = 0; i < n; ++i) {
            Lu[i] = 2.0 * u[i] - (i > 0 ? u[i - 1] : 0.0) - (i + 1 < n ? u[i + 1] : 0.0);
        }
    };

    // Solve A * x = b with zero allocations
    auto result = num::kernel::raw::cg(
        apply_laplacian,
        x.data(),
        b.data(),
        n,
        work.data(),
        /*tol=*/1e-10,
        /*max_iter=*/2000
    );

    std::cout << "CG Converged: " << std::boolalpha << result.converged
              << " in " << result.iterations << " iterations, residual: "
              << result.residual << "\n";
}
```

---

## 4. Higher Tiers and the Single Implementation Principle

Every high-level algorithm in Numerics lowers directly to the Tier-0 raw kernels. There is **exactly one mathematical implementation** for each algorithm in the codebase:

1. **Typed High-Level Layer (`num::cg`, `num::cholesky`):** Enforces C++20 concepts (`SPDOperator`, `NormedSpace`), validates invariants under active diagnostic presets, extracts buffer pointers, and dispatches to the raw kernel.
2. **Container Adaptors (`num::Matrix`, `num::Vector`):** Manage contiguous memory lifetimes and provide convenient algebraic syntax.
3. **Hardware Dispatch:** When BLAS/LAPACK backends are linked, typed wrappers route large matrix operations to vendor GEMM/POTRF microkernels while small matrices and matrix-free operators execute in-tree Tier-0 code.

---

## 5. Architectural Invariants

The codebase enforces strict structural rules verified during continuous integration:

1. **Standalone Purity:** `kernel` includes nothing outside the C++ standard library.
2. **Zero Core Dependencies:** `core` depends only on `kernel`.
3. **Strict Layer Ordering:** No module may include or depend on any module above it in the tier hierarchy.
4. **Single Definition Rule:** Every concept has exactly one definition site.
5. **No Allocation in Inner Loops:** Solvers and raw kernels never allocate heap memory. All buffers are caller-provided or pre-allocated.

