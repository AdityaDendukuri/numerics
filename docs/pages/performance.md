# Performance and Report Workflow {#page_performance}

This page documents the benchmark/report workflow and the implementation files
used by the dense kernels.

## Configure Benchmarks

```bash
cmake -B build \
  -DNUMERICS_BUILD_TESTS=ON \
  -DNUMERICS_BUILD_BENCHMARKS=ON \
  -DNUMERICS_BUILD_REPORT=ON
cmake --build build -j$(nproc)
```

## Run Focused Benchmarks

```bash
./build/benchmarks/numerics_bench --benchmark_filter=BM_Matmul
./build/benchmarks/numerics_bench --benchmark_filter=BM_Matvec
./build/benchmarks/numerics_bench --benchmark_filter="BM_Dot|BM_Axpy"
./build/benchmarks/numerics_bench --benchmark_filter="BM_LU|BM_QR|BM_SVD"
```

## Generate HTML Report

```bash
cmake --build build --target report
```

The report includes benchmark tables, plots, compiler metadata, backend
detection, and test counts.

## Dense Kernel Files

```text
src/core/backends/seq/matrix.cpp      portable seq and blocked kernels
src/core/backends/opt/matrix.cpp      custom optimized/SIMD kernels
src/core/backends/blas/matrix.cpp     BLAS dgemm/dgemv/daxpy paths
src/core/backends/omp/matrix.cpp      OpenMP wrappers
src/core/backends/gpu/matrix.cpp      CUDA-selected path/fallback
include/kernel/raw.hpp                backend-free raw pointer loops
```

## Compare Backends

```cpp
for (num::Backend b : {num::seq, num::blocked, num::simd, num::blas, num::omp}) {
    num::matmul(A, B, C, b);
    check_residual(A, B, C);
}
```

## Implementation Boundary

`kernel::raw` is the scalar primitive layer and does not call BLAS. BLAS calls
are confined to `src/core/backends/blas`. This makes backend comparisons
explicit and keeps the custom kernels independently testable.
