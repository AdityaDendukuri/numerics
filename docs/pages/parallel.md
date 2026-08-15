# Parallel, GPU, and MPI Implementation Note {#page_parallel}

Parallel support is optional and selected at configuration time. Public calls
remain available in non-parallel builds through fallback paths or stubs.

## OpenMP

OpenMP is selected by passing `Backend::omp` to routines that document an OpenMP
path.

```cpp
double I = num::trapz(f, 0.0, 1.0, 1000000, num::Backend::omp);
num::matmul(A, B, C, num::Backend::omp);
num::jacobi(A, b, x, 1e-8, 1000, num::Backend::omp);
```

Implementation locations:

```text
src/core/backends/omp/
src/analysis/quadrature.cpp
src/linalg/solvers/jacobi.cpp
```

## CUDA

CUDA vector and matrix entry points use `Backend::gpu`.

```cpp
num::axpy(3.0, x, y, num::Backend::gpu);
double r = num::norm(y, num::Backend::gpu);
num::matvec(A, x, y, num::Backend::gpu);
```

Implementation locations:

```text
include/core/parallel/cuda_ops.hpp
src/core/parallel/cuda_ops.cu
src/core/parallel/cuda_stubs.cpp
src/core/backends/gpu/
```

When CUDA is not enabled, the stub implementation keeps downstream builds
portable.

## GPU Banded Solve

```cpp
auto result = num::banded_solve(A, b, num::Backend::gpu);
```

The GPU path is intended for many structured systems or large banded problems.
For small systems, launch overhead can dominate.

## MPI

MPI helpers are exposed under `num::mpi`.

```cpp
int rank = num::mpi::rank();
int size = num::mpi::size();

double total = num::mpi::allreduce_sum(local_value);
```

Implementation locations:

```text
include/core/parallel/mpi_ops.hpp
src/core/parallel/mpi_ops.cpp
src/core/parallel/mpi_stubs.cpp
```

## Backend Boundaries

Raw kernels do not call CUDA, MPI, OpenMP, or BLAS. Backend-specific calls live
under `src/core/backends` or `src/core/parallel`.
