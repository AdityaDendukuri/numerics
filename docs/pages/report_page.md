# Benchmark & Performance Report {#page_report}

Numerics features an automated, data-driven benchmark and verification pipeline that compiles, executes, measures, and visualizes performance across the entire library against industry-standard reference libraries (CBLAS, LAPACKE, FFTW3, OpenMP).

---

## 1. Accessing the Generated Reports

The benchmark suite automatically generates both standalone HTML and Markdown reports alongside publication-quality PNG diagnostic plots:

* **Interactive HTML Report:** [Open docs/report/index.html](report/index.html) (self-contained document with embedded styles, benchmark tables, and convergence curves)
* **Markdown Report:** [`docs/report/REPORT.md`](report/REPORT.md)
* **Generated Diagnostic Plots:** Available under `docs/report/plots/`

---

## 2. Generating the Report Locally

The entire benchmark matrix, test verification, and plot generation pipeline is automated via the CMake `report` target:

```bash
# 1. Configure with tests, benchmarks, and report generator enabled
cmake -B build -S . \
  -DNUMERICS_BUILD_TESTS=ON \
  -DNUMERICS_BUILD_BENCHMARKS=ON \
  -DNUMERICS_BUILD_REPORT=ON \
  -DCMAKE_BUILD_TYPE=Release

# 2. Execute tests, Google Benchmark suites, convergence diagnostics, and assemble report
cmake --build build --target report
```

### What the report Target Executes:
1. **Unit Test Verification:** Runs all 239 Google Test cases via `numerics_tests` and serializes `output/test_results.json`.
2. **Microbenchmarks (Google Benchmark):** Executes `numerics_bench`, profiling native in-tree C++20 routines against hardware-accelerated backends (`output/bench_results.json`).
3. **Convergence Diagnostics:** Executes `bench_convergence` to generate convergence histories on 2D PDE problems, MINRES vs. CG energy monotonicity, Talbot contour resolvent solves, and long-term symplectic ODE Hamiltonian preservation (`output/plots/*.png`).
4. **Markdown & HTML Report Assembly:** Compiles and runs `gen_report` (`cmake/report/gen_report.cpp`), which parses JSON results, renders comparison markdown tables, generates `docs/report/REPORT.md`, and bundles an interactive `docs/report/index.html`.

---

## 3. Empirical Comparison: In-Tree C++20 vs. Vendor BLAS/LAPACK

The automated benchmark suite profiles both execution paths side-by-side:

### Empirical Findings from Benchmark Data

| Benchmark Scope | In-Tree C++20 Kernels (`kernel::raw`) | Hardware Backend (BLAS / LAPACKE / FFTW3) | Empirical Comparison & Tradeoff |
| :--- | :--- | :--- | :--- |
| **BLAS-1 / Vector Operations** (`dot`, `axpy`, `norm`) | Streamlined register loops (`kernel::raw::axpy`, `dot`) | `cblas_daxpy`, `cblas_ddot` (Apple Accelerate / OpenBLAS) | Native in-tree scalar/SIMD kernels match or exceed BLAS throughput on small-to-medium vectors (\f$N \le 4096\f$) by eliminating library call overhead; OpenMP parallel loops dominate for large vectors (\f$N \ge 262144\f$). |
| **Matrix-Vector Multiply** (`matvec`) | Row-major tiled loop (`kernel::raw::matvec`) | `cblas_dgemv` | Hardware CBLAS achieves peak memory bandwidth utilizing SIMD registers (up to \f$160+\text{ GB/s}\f$ on unified memory architectures). |
| **Matrix Multiplication** (`gemm`) | Portable blocked loops (`Matmul<backend::blocked>`) | `cblas_dgemm` | Vendor GEMM microkernels achieve peak GFLOPS through hardware-specific matrix accelerators (e.g. Apple AMX / AVX-512 FMA). |
| **Dense Factorizations** (LU / QR / Cholesky) | Standalone in-place factorization (`kernel::raw::lu_factor`, `cholesky`) | `LAPACKE_dgetrf`, `LAPACKE_dgeqrf`, `LAPACKE_dpotrf` | For small matrices (\f$N \le 64\f$), in-tree unblocked kernels execute with minimal latency; for large matrices (\f$N \ge 512\f$), LAPACK's blocked BLAS-3 updates scale efficiently. |
| **Eigensolvers & SVD** | Cyclic Jacobi (`EigSym<backend::seq>`), One-Sided Jacobi SVD | `LAPACKE_dsyevd`, `LAPACKE_dgesdd` (Divide-and-Conquer) | LAPACK divide-and-conquer provides optimal \f$\mathcal{O}(N^3)\f$ scaling for large dense matrices; native matrix-free **Lanczos** (`num::lanczos`) and **Randomized SVD** (`num::randomized_svd`) excel when computing a low-rank subspace (\f$k \ll N\f$). |
| **Fast Fourier Transforms** | Cooley-Tukey & NEON SIMD (`spectral::fft`) | `FFTW3` (`fftw_plan_dft_1d`) | FFTW3 achieves peak throughput via pre-computed execution plans; native in-tree FFT provides zero-dependency standalone execution. |

---

## 4. Supercomputing Environments & Compiler Portability

Numerics interfaces with vendor-optimized math libraries strictly through standardized C-ABIs (**CBLAS** and **LAPACKE**):
* **AMD HPC (EPYC / Instinct):** AMD Optimizing CPU Libraries (AOCL / BLIS / libFLAME) via AOCC or GCC.
* **Intel HPC (Xeon / Max GPUs):** Intel oneAPI MKL via `icpx` / `icx`.
* **ARM HPC (Graviton / Fugaku / Apple Silicon):** Arm Performance Libraries (ARMPL), Fujitsu SSL2, or Apple Accelerate via Clang.
* **Cray HPE Systems:** Cray LibSci via Cray Compiling Environment (`craycc`).

If a target system lacks external math libraries, Numerics compiles seamlessly with its standalone in-tree C++20 kernels with zero missing dependencies.

