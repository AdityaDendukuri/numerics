# Backend Namespaces, Parallelism, & Hardware Acceleration {#page_parallel}

Optional hardware acceleration across BLAS/LAPACK, OpenMP, CUDA, MPI, and FFTW. Compiles to portable standard C++20 by default with compile-time capability detection.

Vectorization is not a backend and cannot be selected. `num::kernel` contains no intrinsics and performs no runtime CPU dispatch. It writes loops the compiler can vectorize, and blocks them for the register file and the cache. Every backend below calls into it, so all of them benefit.

```cpp
#include <numerics.hpp>
```

---

## 1. Backends Are Plain Namespaces

There is no tag type, no enum, and no runtime dispatch switch. Every backend is a
namespace of free functions with the same signatures as `num::kernel`'s:

```cpp
namespace num {
namespace kernel {} // The dependency-free reference: raw pointers only, no vec/mat.
namespace seq {}    // vec/mat-aware wrapper over num::kernel. Always available.
namespace omp {}    // OpenMP parallel loops. Falls back to num::kernel when
                    // NUMERICS_HAS_OMP is not defined. Always compiles.
namespace blas {}   // cblas_* calls. Falls back to num::kernel when
                    // NUMERICS_HAS_BLAS is not defined. Always compiles.
namespace lapack {} // LAPACKE calls. Falls back to num::seq when
                    // NUMERICS_HAS_LAPACK is not defined. Always compiles.
namespace cuda {}   // CUDA device kernels. Throws std::runtime_error when called
                    // on a build without NUMERICS_HAS_CUDA. See §4.
}
```

Call a backend by name. The choice resolves at compile time and inlines into the
caller.

```cpp
num::vec x(n, 1.0), y(n, 2.0);

num::real a = num::dot(x, y);        // Build default: see num::accel below.
num::real b = num::seq::dot(x, y);   // Portable loop, forced.
num::real c = num::blas::dot(x, y);  // cblas_ddot, forced (falls back to seq if
                                     // BLAS wasn't configured).
```

Code written against any backend namespace compiles in every build. The namespace
always exists, and its functions fall back to `num::kernel` when the underlying library
was not configured. CUDA is the exception; see §4.

### Compile-Time Feature Detection Flags

| Feature Flag | C++ Compile-Time Constant | Target Hardware / Library |
| :--- | :--- | :--- |
| `NUMERICS_HAS_BLAS` | `num::has_blas` | OpenBLAS, Apple Accelerate, MKL |
| `NUMERICS_HAS_LAPACK` | `num::has_lapack` | LAPACKE C interface |
| `NUMERICS_HAS_OMP` | `num::has_omp` | Multi-core OpenMP thread pools |
| `NUMERICS_HAS_SIMD` | `num::has_simd` | Compiler was given AVX-256 + FMA (x86-64); NEON is baseline on AArch64. Read only by the FFT's intrinsic path. `num::kernel` needs no flag. |
| `NUMERICS_HAS_CUDA` | `num::has_cuda` | NVIDIA GPU CUDA acceleration |
| `NUMERICS_HAS_MPI` | (none) | Multi-node distributed memory |
| `NUMERICS_HAS_FFTW` | `num::spectral::has_fftw` | FFTW3 optimized Fourier engine |

These flags are attached only to the CMake target for that backend
(`numerics::blas`, `numerics::omp`, and so on), not globally. See
@ref page_architecture "the architecture page" for how `numerics::kernel` and
`numerics::core` stay dependency-free when every backend is present on the configuring
machine.

---

## 2. `num::accel`: the Untagged Default

Routines called without naming a backend (`num::dot`, `num::axpy`, `num::scale`,
`num::norm`, `num::add`, `num::matvec`, `num::matmul`) resolve through a single
namespace alias, chosen once at configure time in `core/policy.hpp`:

```cpp
#if defined(NUMERICS_HAS_CUDA)
namespace accel = cuda;
#elif defined(NUMERICS_HAS_BLAS)
namespace accel = blas;
#elif defined(NUMERICS_HAS_OMP)
namespace accel = omp;
#else
namespace accel = seq;
#endif
```

`num::dot(x, y)` is defined as `accel::dot(x, y)`. There is no runtime branch. The
preprocessor resolves the alias before the translation unit is parsed.

### Factorizations pick their own default

Factorizations (`lu`, `qr`, `svd`, `eig_sym`, `hessenberg`, `thomas`) are not
routed through `num::accel`. LAPACK is a poor substitute for BLAS or OpenMP on level-1
and level-2 operations, and the reverse also holds, so each factorization chooses
independently.

```cpp
inline lu_result lu(const linear::sq_mat<mat> &A) {
#if defined(NUMERICS_HAS_LAPACK)
    return lapack::lu(A.base());
#else
    return seq::lu(A.base());
#endif
}
```

To bypass this and force a specific implementation, call the namespace directly:
`num::lapack::lu(A)`, `num::seq::lu(A)`, `num::lapack::eig_sym(A)`,
`num::seq::svd(A, tol, max_sweeps)`, and so on. Each factorization's header documents
the pair it offers.

### Spectral Resolution (`num::spectral::default_fft_backend`)

FFT keeps its own separate resolution and its own `fft_backend` enum (`fftw` →
`simd`/`std::simd` → `seq`), unrelated to `num::accel`. See the FFT plan API in
`spectral/fft.hpp`.

---

## 3. Worked Example: Switching One Call from `kernel` to LAPACK/CUDA

`num::kernel` is the reference implementation every backend is checked against. To use a
faster backend at one call site, name that backend's namespace.

```cpp
// Portable reference. Always compiles, no external library.
num::lu_result ref = num::seq::lu(A);

// Same algorithm through vendor LAPACK when NUMERICS_HAS_LAPACK is set.
// Falls back to num::seq::lu otherwise, so this line never needs an #ifdef.
num::lu_result fast = num::lapack::lu(A);

// Let the build decide: LAPACK if configured, else seq. The common case.
num::lu_result f = num::lu(num::assume_square(A));

// Force OpenMP for a vector op:
num::omp::axpy(2.0, x, y);

// Force BLAS for a vector op:
num::blas::axpy(2.0, x, y);
```

To swap which backend a whole build defaults to, change what's linked, not the
call sites: link `numerics::blas`/`numerics::omp`/`numerics::cuda` (see
@ref page_architecture) and `num::accel` re-resolves at the next compile.

---

## 4. CUDA GPU Acceleration

`num::cuda`'s host-container overloads (`num::cuda::scale(vec&, real)`,
`num::cuda::axpy(...)`, `num::cuda::dot(...)`, `num::cuda::matvec(...)`,
`num::cuda::matmul(...)`) sit over a raw device-pointer API
(`num::cuda::scale(real*, idx, real)`, ...) meant for callers who manage device
buffers directly across a whole algorithm. See `unsafe::cuda::cg` in
`linear/solvers/cg.hpp` for the pattern.

Unlike `omp`/`blas`/`lapack`, **`num::cuda` does not silently degrade**: on a
build without `NUMERICS_HAS_CUDA`, every function in the namespace throws
`std::runtime_error("CUDA not available")` if actually called. This is
deliberate. A GPU call that silently ran on the CPU would produce a correct result and a
misleading measurement. Guard
CUDA-specific code with `#if defined(NUMERICS_HAS_CUDA)` (or `num::has_cuda`)
if it must also build without a device toolkit:

```cpp
#if defined(NUMERICS_HAS_CUDA)
x.to_gpu();
y.to_gpu();
num::cuda::axpy(3.0, x, y);
double r = num::cuda::norm(y);
#endif
```

---

## 5. Distributed Memory via MPI (`num::mpi`)

MPI communication helpers are exposed under `num::mpi`, built only into the
separate `numerics::mpi` target so linking `numerics::core`/`numerics::numerics`
never pulls in an MPI dependency:

```cpp
int rank = num::mpi::rank();
int size = num::mpi::size();

// Collective reduction
double global_sum = num::mpi::allreduce_sum(local_value);
```
