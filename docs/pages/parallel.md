# Backend Resolution, Parallelism, & Hardware Acceleration {#page_parallel}

Multi-tier hardware acceleration across BLAS/LAPACK, OpenMP, SIMD vectorization, CUDA, MPI, and FFTW. Compiles to portable standard C++20 by default with automatic compile-time capability detection.

```cpp
#include <numerics.hpp>
```


---

## 1. Backend Tags

A backend is selected by passing a tag. Tags are empty types, so the selection is resolved during compilation and the chosen kernel inlines into the caller:

```cpp
namespace num::backend {
struct seq_t;     // Portable scalar C++ loop.
struct blocked_t; // Cache-blocked CPU loop.
struct simd_t;    // Explicit AVX2 / ARM NEON intrinsics.
struct blas_t;    // Vendor BLAS (OpenBLAS, Apple Accelerate, Intel MKL).
struct omp_t;     // OpenMP parallel loops.
struct lapack_t;  // Vendor LAPACK (LAPACKE).
struct gpu_t;     // CUDA device kernels.
}
```

Each tag has a constant of the same name without the suffix:

```cpp
num::Vector x(n, 1.0), y(n, 2.0);

num::real a = num::dot(x, y);                    // Build default.
num::real b = num::dot(x, y, num::backend::seq); // Portable loop.
num::real c = num::dot(x, y, num::backend::blas);// cblas_ddot.
```

A tag whose backend was not detected at configure time falls back to `seq`. Code written against any tag compiles and runs everywhere.

### Compile-Time Feature Detection Flags


| Feature Flag | C++ Compile-Time Constant | Target Hardware / Library |
| :--- | :--- | :--- |
| `NUMERICS_HAS_BLAS` | `num::has_blas` | OpenBLAS, Apple Accelerate, MKL |
| `NUMERICS_HAS_LAPACK` | `num::has_lapack` | LAPACKE C interface |
| `NUMERICS_HAS_OMP` | `num::has_omp` | Multi-core OpenMP thread pools |
| `NUMERICS_HAS_SIMD` | `num::has_simd` | AVX-256, FMA, ARM NEON |
| `NUMERICS_HAS_CUDA` | `num::has_cuda` | NVIDIA GPU CUDA acceleration |
| `NUMERICS_HAS_MPI` | (none) | Multi-node distributed memory |
| `NUMERICS_HAS_FFTW` | `num::spectral::has_fftw` | FFTW3 optimized Fourier engine |

---

## 2. Backend Resolution

Routines called without a tag use the strongest backend the build detected.

### Dense Level-1 and Level-2 (num::backend::dflt)

\f[
\text{BLAS} \;\longrightarrow\; \text{OpenMP} \;\longrightarrow\; \text{SIMD} \;\longrightarrow\; \text{Cache-Blocked Sequential}
\f]

```cpp
using default_t =
#if defined(NUMERICS_HAS_BLAS)
    blas_t;
#elif defined(NUMERICS_HAS_OMP)
    omp_t;
#elif defined(NUMERICS_HAS_SIMD)
    simd_t;
#else
    blocked_t;
#endif
```

### Factorizations (num::backend::factor)

\f[
\text{LAPACK (LAPACKE)} \;\longrightarrow\; \text{OpenMP} \;\longrightarrow\; \text{Inlined C++ (kernel::raw)}
\f]

```cpp
using factor_t =
#if defined(NUMERICS_HAS_LAPACK)
    lapack_t;
#elif defined(NUMERICS_HAS_OMP)
    omp_t;
#else
    seq_t;
#endif
```

### Spectral Resolution (num::spectral::default_fft_backend)

\f[
\text{FFTW3} \;\longrightarrow\; \text{SIMD Radix-2} \;\longrightarrow\; \text{Scalar Radix-2 Cooley–Tukey}
\f]

---

### Selecting a Backend at Run Time

`num::Backend` is an enum of the same alternatives, for values that are not known until run time. `num::with_backend` converts one into a tag:

```cpp
num::Backend chosen = parse_backend(argv[1]);

num::with_backend(chosen, [&](auto tag) {
    num::matvec(A, x, y, tag); // Compiles one instantiation per alternative.
});
```

The switch runs once. Inside the lambda the tag is a compile-time type again, so the kernel inlines as it would with a literal tag.

## 3. Explicit Backend Selection & Graceful Degradation

Every major algorithm (e.g. `num::lu`, `num::cholesky`, `num::matmul`, `num::cg`, `num::trapz`) allows explicit backend overrides at the call site:

```cpp
// 1. Force vendor LAPACK / BLAS
num::LUResult factor = num::lu(num::assume_square(A), num::backend::lapack);
num::matmul(A, B, C, num::backend::blas);

// 2. Force OpenMP multi-core parallelization
double integral = num::trapz(f, 0.0, 1.0, 1000000, num::backend::omp);
num::cg(Aop, b, x, 1e-8, 1000, num::backend::omp);

// 3. Force pure sequential reference path (for debugging / verification)
num::LUResult ref_factor = num::lu(num::assume_square(A), num::backend::seq);
```

### Graceful Fallback Guarantee

If user code requests a backend that was not compiled into the binary (e.g., passing `num::backend::lapack` on a build without LAPACK, or `num::backend::gpu` without CUDA):
* The dispatcher **never throws a missing-symbol or segmentation fault error**.
* It automatically falls back to the best available CPU implementation (`seq` or `blocked`) and logs a diagnostic when runtime checking is active.

---

## 4. OpenMP Policy Dispatch (num::seq vs num::par)

For container-level vector and matrix kernels, policy tags provide zero-overhead compile-time dispatch:

```cpp
// Explicit sequential execution
num::container/array.hpp::axpby(2.0, x, 0.5, y, num::seq);

// Explicit OpenMP parallel execution
num::container/array.hpp::axpby(2.0, x, 0.5, y, num::par);
```

---

## 5. CUDA GPU Acceleration

GPU entry points operate on host containers or device buffers:

```cpp
num::axpy(3.0, x, y, num::backend::gpu);
double r = num::norm(y, num::backend::gpu);
num::matvec(A, x, y, num::backend::gpu);
```

---

## 6. Distributed Memory via MPI (num::mpi)

MPI communication helpers are exposed under `num::mpi`:

```cpp
int rank = num::mpi::rank();
int size = num::mpi::size();

// Collective reduction
double global_sum = num::mpi::allreduce_sum(local_value);
```
