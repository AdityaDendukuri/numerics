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
# Matrix Multiplication (kernel vs OpenMP vs BLAS)
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
include/kernel/{vector,dense,sparse,rotations,factor,krylov}.hpp   Zero-allocation, inlined raw-pointer loops
include/seq/            (in container/{vector,matrix,dense,reduce}_ops.hpp)  vec-aware wrappers over num::kernel
include/omp/{vector_ops,matrix_ops,parallel_ops}.hpp                OpenMP-parallel loops
include/blas/{vector_ops,matrix_ops}.hpp                            cblas_* wrappers
include/cuda/{cuda_ops,container_ops}.hpp                           CUDA device kernels
include/lapack/lapack_wrapper.hpp                                   LAPACKE wrappers
```

### Separation of Concerns

* **`num::kernel`** contains pure arithmetic loops over raw pointers and does not call external libraries or know about `vec`/`mat`.
* It carries **no intrinsics and no runtime CPU dispatch.** Vectorization is the compiler's job; the kernel's job is to write loops it can vectorize, and to block them for the register file and the cache. `kernel::gemm` sizes its register tile from `NUM_K_VECTOR_REGISTERS` and its cache panel from a byte budget, both at compile time. Hand-written AVX2 and NEON products lived here once and were removed: the portable tiled `gemm` measured 30.0 GFLOP/s against their 23.7 on the same machine, and the intrinsic versions had a leading-dimension bug that made them silently wrong for any non-square shape.
* Every accelerator is a plain namespace of free functions matching `num::kernel`'s signatures: `num::seq` (the `vec`/`mat`-aware fallback), `num::omp`, `num::blas`, `num::cuda`. There is no tag or enum layer between a caller and these — call a backend by name (`num::omp::dot(x, y)`), or let the untagged `num::dot(x, y)` resolve through `num::accel`, the single compile-time default (CUDA > BLAS > OMP > seq, whichever was configured).
* This isolation guarantees that custom algorithms can be benchmarked against vendor BLAS/LAPACK cleanly and reproducibly.

---

## 5. Backend Comparison

```cpp
#include <numerics.hpp>

// Compare wall-clock timing and correctness across every configured backend:
num::seq::matmul(A, B, C);
check_residual(A, B, C);
#if defined(NUMERICS_HAS_BLAS)
num::blas::matmul(A, B, C);
check_residual(A, B, C);
#endif
#if defined(NUMERICS_HAS_OMP)
num::omp::matmul(A, B, C);
check_residual(A, B, C);
#endif
```
