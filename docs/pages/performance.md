# Performance Benchmarks & Report Workflow {#page_performance}

This page documents the benchmark workflow, automated performance reporting, and dense kernel architecture.

---

## 1. Configure and Build Benchmarks

```bash
cmake -B build \
  -DNUMERICS_BUILD_TESTS=ON \
  -DNUMERICS_BUILD_BENCHMARKS=ON \
  -DNUMERICS_BUILD_REPORT=ON
cmake --build build -j$(nproc)
```

---

## 2. Run Focused Microbenchmarks

Google Benchmark targets measure throughput, cache scalability, and acceleration speedup:

```bash
# Matrix Multiplication (Seq vs Blocked vs SIMD vs BLAS vs OpenMP)
./build/benchmarks/numerics_bench --benchmark_filter=BM_Matmul

# Matrix-Vector Multiplication & Level-1 BLAS
./build/benchmarks/numerics_bench --benchmark_filter=BM_Matvec
./build/benchmarks/numerics_bench --benchmark_filter="BM_Dot|BM_Axpy"

# Factorizations & Spectral Solvers
./build/benchmarks/numerics_bench --benchmark_filter="BM_LU|BM_QR|BM_SVD|BM_Hessenberg"
```

---

## 3. Automated HTML Performance Report

Generate the complete HTML benchmark report with interactive plots and metadata:

```bash
cmake --build build --target report
```

The report includes:
* **Benchmark Tables:** GFLOP/s, execution times, and speedup ratios.
* **Convergence Plots:** Iterative Krylov residuals, symplectic energy conservation, and Talbot contour quadrature decay.
* **Hardware & Compiler Metadata:** CPU architecture, vector extension flags, cache line sizes, and active backends.

---

## 4. Kernel Architecture & Implementation Boundaries

```text
src/container/backends/seq/matrix.cpp     Portable sequential and cache-blocked kernels
src/container/backends/opt/matrix.cpp     Explicit SIMD vectorization kernels (AVX2 / NEON)
src/container/backends/blas/matrix.cpp    Vendor BLAS dgemm/dgemv/daxpy wrappers
src/container/backends/omp/matrix.cpp     Multi-threaded OpenMP parallel wrappers
src/container/backends/gpu/matrix.cpp     CUDA GPU execution paths
include/kernel/raw.hpp               Zero-allocation, inlined raw-pointer loops
```

### Separation of Concerns

* **`kernel::raw`** contains pure scalar arithmetic loops and does not call external libraries (BLAS/LAPACK/CUDA).
* **Vendor acceleration calls** (OpenBLAS, LAPACKE, CUDA) are isolated in `src/container/backends/` and `src/linear/factorization/`.
* This isolation guarantees that custom algorithms can be benchmarked against vendor BLAS/LAPACK cleanly and reproducibly.

---

## 5. Runtime Backend Comparison

```cpp
#include <numerics.hpp>

// Verify correctness and compare wall-clock timing across all active backends:
for (num::Backend b : {num::seq, num::blocked, num::simd, num::blas, num::omp}) {
    num::matmul(A, B, C, b);
    check_residual(A, B, C);
}
```
