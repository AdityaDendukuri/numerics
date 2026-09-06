# Library Structure & Architecture {#page_architecture}

Numerics is organized into unidirectional tiers. Each tier depends only on the tiers below it. Any tier can therefore be included, tested, or copied into another codebase on its own. This matters most for the kernel tier, which has no dependencies at all.

---

## 1. Architectural Tiers

```
┌─────────────────────────────────────────────────────────────┐
│ Tier 4: Problem Dispatch (num::solve, linear_problem)        │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│ Tier 3: Domain Modules (linear, ode, pde, spectral, etc.)   │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│ Tier 2: Containers & Operators (vec, mat, dense_op)    │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│ Tier 1: Concepts & Invariants (algebra, core, diagnostics)  │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│ Tier 0: Standalone Raw Compute Kernels (num::kernel)         │
└─────────────────────────────────────────────────────────────┘

   ── beside the stack, not inside it ──────────────────────────
   include/{omp,blas,lapack,cuda,mpi}/   Accelerators: each one a
                                         plain namespace whose
                                         functions call an external
                                         library, or funnel back
                                         into num::kernel.
   include/{io,plot}/                    Header-only utilities;
                                         nothing in the stack
                                         depends on them.
```

The accelerators sit *beside* the tier stack rather than at a level in it. Each
is a top-level directory (`include/omp/`, `include/blas/`, ...) holding one
namespace of free functions with `num::kernel`'s signatures. They are not a tier
because nothing in the stack is built on them: they are optional substitutions
for leaf kernels, chosen by what the build links. See
@ref page_parallel "Backend Namespaces & Hardware Acceleration" for how a call
site picks one.

| Tier | Module | Responsibilities | Dependencies |
| :--- | :--- | :--- | :--- |
| **Tier 0** | `kernel` | Raw compute over pointers (`T*`), dimensions, and callables; zero allocations | *None* (Pure Standard C++20) |
| **Tier 1** | `core`, `algebra` | scalar fields, vector spaces, property hierarchy, and runtime diagnostic evidence | `kernel` |
| **Tier 2** | `container`, `operator` | `vec`, `mat`, `spmat`, matrix-free operators; defines `num::seq` (container-aware wrapper over `num::kernel`) and the untagged `num::` entry points that resolve through `num::accel` | `core`, `algebra`, `kernel` |
| **Tier 3** | `linear`, `ode`, `pde`, `spectral`, `quadrature`, `roots`, `structures`, `spatial`, `stochastic` | Numerical domain algorithms (factorizations, Krylov solvers, RK45, Verlet, FFT, graph structures) | Tier 0–2 |
| **Tier 4** | `solve` | Unified problem dispatch (`linear_problem`, `ode_problem`) | All tiers |
| **Accelerators** | `omp`, `blas`, `lapack`, `cuda`, `mpi` | One namespace each (`num::omp`, `num::blas`, ...), same signatures as `num::kernel`. Optional; selected by what the build links, never by a tag or enum | `container`, plus the external library |
| **Auxiliary** | `io`, `plot` | Header-only I/O and terminal plotting. Nothing in Tiers 0–4 includes them, so they can be deleted or lifted out on their own | `container` |

---

## 2. The Standalone Raw Compute Layer (num::kernel)

Per-routine documentation is in @ref page_kernel "the num::kernel reference".

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
* `<kernel/vector.hpp>`: BLAS-1 vector kernels and the fused/reduction primitives the rest of the tier builds on (macros, `contract` tags, `detail::reduce`).
* `<kernel/dense.hpp>`: BLAS-2/3 dense matrix kernels, triangular solves (`trsv`, `trsm`), GEMM, banded ops, and Gram–Schmidt orthogonalization.
* `<kernel/sparse.hpp>`: CSR SpMV/SpMM and ILU(0) factorization.
* `<kernel/rotations.hpp>`: Givens, Householder, and Jacobi rotations, and blocked QR.
* `<kernel/factor.hpp>`: Cholesky ($A = LL^T$), blocked Cholesky, LU with partial pivoting ($PA = LU$), blocked LU, shifted Hessenberg factorizations, and banded LU solvers.
* `<kernel/krylov.hpp>`: mat-free Conjugate Gradient (`cg`) and Preconditioned Conjugate Gradient (`pcg`) operating on generic callable operators.
* `<kernel/complex.hpp>`: The routines that mix real and complex operands (real matrix times complex vector, shifted Hessenberg solves). Kept separate because `<complex>` costs ~95k preprocessed lines; the umbrella includes it, a direct include of `dense.hpp` does not.
* `<kernel/debug.hpp>`: `operator<<` for `krylov_result`. This is the only kernel header that includes `<ostream>`, and it is **not** part of the umbrella. The compute path therefore never pulls in iostreams, and a freestanding target without them still compiles.

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
#include <kernel/kernel.hpp>

int main() {
    constexpr num::idx n = 4;
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> y = {0.5, 1.5, 2.5, 3.5};
    std::vector<double> z(n, 0.0);

    // 1. vec linear combination: z = 2.0 * x + 3.0 * y
    num::kernel::axpbyz(z.data(), x.data(), y.data(), 2.0, 3.0, n);

    // 2. Dot product: s = x . y
    double dot_val = num::kernel::dot(x.data(), y.data(), n);

    // 3. Euclidean norm: ||x||_2
    double norm_val = num::kernel::norm(x.data(), n);

    // 4. Dense matrix-vector product: y = A * x (A is 4x4 row-major)
    std::vector<double> A = {
        4.0, 1.0, 0.0, 0.0,
        1.0, 4.0, 1.0, 0.0,
        0.0, 1.0, 4.0, 1.0,
        0.0, 0.0, 1.0, 4.0
    };
    std::vector<double> Ax(n, 0.0);
    num::kernel::matvec(Ax.data(), A.data(), x.data(), /*m=*/n, /*n=*/n);

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
    bool spd = num::kernel::cholesky(L.data(), A.data(), n);
    if (spd) {
        // Solves A * x = b via forward/back substitution through L
        num::kernel::cholesky_solve(x.data(), L.data(), b.data(), n);
        std::cout << "Cholesky Solution: [" << x[0] << ", " << x[1] << ", " << x[2] << "]\n";
    }

    // 2. General LU Factorization with partial pivoting: P * A = L * U
    std::vector<double> LU = A; // Factorization is computed in-place
    std::vector<num::idx> ipiv(n, 0);
    bool nonsingular = num::kernel::lu_factor(LU.data(), ipiv.data(), n);
    if (nonsingular) {
        num::kernel::lu_solve(x.data(), LU.data(), ipiv.data(), b.data(), n);
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

    // mat-free 1D discrete Laplacian operator: y = -u''(x)
    auto apply_laplacian = [n](const double *u, double *Lu) {
        for (num::idx i = 0; i < n; ++i) {
            Lu[i] = 2.0 * u[i] - (i > 0 ? u[i - 1] : 0.0) - (i + 1 < n ? u[i + 1] : 0.0);
        }
    };

    // Solve A * x = b with zero allocations
    auto result = num::kernel::cg(
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

1. **Typed High-Level Layer (`num::cg`, `num::cholesky`):** Enforces C++20 concepts (`spd_operator`, `normed_space`), validates invariants under active diagnostic presets, extracts buffer pointers, and dispatches to the raw kernel.
2. **Container Adaptors (`num::mat`, `num::vec`):** Manage contiguous memory lifetimes and provide convenient algebraic syntax.
3. **Hardware Dispatch:** When BLAS/LAPACK backends are linked, typed wrappers route large matrix operations to vendor GEMM/POTRF microkernels while small matrices and matrix-free operators execute in-tree Tier-0 code.

---

## 5. Where a New Feature Goes

The tier a thing belongs to is decided by one question: **what does it need to
know about?**

| You are adding | It goes in | Because |
| :--- | :--- | :--- |
| A loop over `T*` and lengths: a new BLAS-like primitive, a factorization step, a stencil apply | `include/kernel/<area>.hpp` | It knows only pointers. Nothing above Tier 0 may be mentioned: no `vec`, no `mat`, no external library, no allocation. |
| A new concept or property tag (`spd_operator`, `normed_space`) | `include/algebra/` | Tier 1 is where mathematical structure is *stated*; Tier 0 is where it is *computed*. |
| A container operation on `vec`/`mat` | `include/container/<name>_ops.hpp` | Define it in `num::seq` (the portable path) and, if it needs one, an untagged `num::` forward that resolves through `num::accel`. |
| A numerical algorithm: a solver, an integrator, a transform | `include/<domain>/` (`linear`, `ode`, `pde`, `spectral`, ...) | Tier 3. It composes containers and concepts; it must not re-implement arithmetic that belongs in `kernel`. |
| A faster path using an external library | `include/<backend>/<area>_ops.hpp` | A new sibling namespace, matching `num::kernel`'s signatures exactly. Add a CMake target that carries the `NUMERICS_HAS_<X>` define, and give it a `num::seq` fallback so the namespace always compiles. |

### The rule that decides the hard cases

**Every computation that does not call an external library must reach
`num::kernel`.** A domain module that writes its own scalar loop instead of
calling a kernel primitive has created a second implementation of that
arithmetic, and the two will drift. If the primitive you need is missing, add it
to `kernel` and call it. Do not inline it upstairs.

The arrangement follows that of an operating system kernel. `num::kernel` is a small
unchecked layer that runs as close to the hardware as portable C++ allows. The tiers above
it make that layer safe to use. They own the invariants, the dimension checks, the memory
lifetimes, and the diagnostics. The kernel owns the arithmetic and assumes the rest has
already been established.

### What `kernel` may not contain

* **No intrinsics and no runtime CPU dispatch.** Vectorization is the compiler's
  job; the kernel's job is to write loops it can vectorize and to block them for
  the register file and the cache (`kernel::gemm` is the worked example). Hand-
  written AVX2 and NEON products were removed for two reasons. The portable tiled `gemm`
  measured faster, and the intrinsic versions indexed `A` with the wrong leading
  dimension, which made them silently incorrect for non-square shapes.
* **No parallelism.** Threading requires a runtime (OpenMP, CUDA), which is an
  external dependency. Parallel decomposition belongs in `num::omp`/`num::cuda`,
  which slice the problem and call `num::kernel` per block.
* **No allocation, no exceptions, no I/O.**

---

## 6. Architectural Invariants

The codebase enforces strict structural rules verified during continuous integration:

1. **Standalone Purity:** `kernel` includes nothing outside the C++ standard library.
2. **Zero Core Dependencies:** `core` depends only on `kernel`.
3. **Strict Layer Ordering:** No module may include or depend on any module above it in the tier hierarchy.
4. **Single Definition Rule:** Every concept has exactly one definition site.
5. **No Allocation in Inner Loops:** Solvers and raw kernels never allocate heap memory. All buffers are caller-provided or pre-allocated.

