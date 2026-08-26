# Benchmark & Performance Report {#page_report}

Numerics maintains an auto-generated, comprehensive benchmark and numerical validation report that tracks kernel throughput, linear algebra scaling against BLAS/LAPACK/FFTW baselines, iterative solver residual histories, spectral contour quadrature convergence, and symplectic Hamiltonian energy preservation.

---

## Accessing the Report

* **Interactive HTML Report:** [Open docs/report/index.html](report/index.html)
* **Markdown Report Source:** [`docs/report/REPORT.md`](report/REPORT.md)

---

## Generating the Report Locally

The complete test suite, benchmark suite, and convergence diagnostics can be compiled and rendered locally with a single target:

```bash
# Configure with tests and benchmarks enabled
cmake -B build -S . -DNUMERICS_BUILD_TESTS=ON -DNUMERICS_BUILD_BENCHMARKS=ON

# Run test execution, Google Benchmark suites, diagnostic plot generation, and assembly
cmake --build build --target report
```

This updates:
1. `docs/report/REPORT.md` (Markdown status and performance document)
2. `docs/report/index.html` (self-contained HTML report with embedded styles and figures)
3. `docs/report/plots/*.png` (high-resolution SIAM-style benchmark and convergence figures)

---

## Benchmark & Validation Scope

| Domain | Key Benchmarks & Diagnostics | Reference / Baseline |
| :--- | :--- | :--- |
| **BLAS-1 / BLAS-2 / BLAS-3** | Matrix multiply ($2 n^3 / t$), Matvec ($GB/s$), Dot product, Axpy | `cblas_dgemm`, `cblas_dgemv`, `cblas_ddot`, `cblas_daxpy` |
| **Factorizations** | LU with partial pivoting, Householder QR, Thomas tridiagonal | `LAPACKE_dgetrf`, `LAPACKE_dgeqrf`, `LAPACKE_dgtsv` |
| **Iterative Solvers** | Residual reduction $\frac{\|r_k\|}{\|b\|}$ on 2D Poisson ($N=1024$): `CG`, `PCG (Jacobi)`, `GMRES (m=30)`, `MINRES`, `Jacobi`, `Gauss-Seidel` | Exact residual norm history |
| **Resolvent & Exponentials** | Weideman-Talbot shifted contour quadrature for matrix exponential action $e^{t Q} p_0$ ($M = 4 \dots 32$ nodes) | Arnoldi Krylov projection (`expv`) |
| **Banded Systems** | Tridiagonal, Pentadiagonal, general $(KL, KU)$ banded factorizations and solves | $O(N)$ linear scaling verification |
| **Eigensolvers** | Full Cyclic Jacobi vs Divide-and-Conquer `dsyevd`; matrix-free Lanczos ($k=10$) | `LAPACKE_dsyevd` |
| **SVD** | One-sided Jacobi vs Randomized Truncated SVD ($k = n/8$) vs Divide-and-Conquer `dgesdd` | `LAPACKE_dgesdd` |
| **Dynamical Systems / ODEs** | Long-term Hamiltonian energy error $|E(t) - E(0)|$ over $2 \times 10^3$ steps ($h = 0.05$): Euler, RK4, Störmer-Verlet, Yoshida 4th-order | Analytical invariant $H(q, p) = \frac{1}{2} p^2 + \frac{1}{2} q^2$ |
| **Spectral Methods** | 1D/2D Forward FFT, amortized reusable FFT plans, real-to-complex rFFT | FFTW3 (`fftw_plan_dft_1d`) |
| **Discrete Graph & Data Structures** | `num::DisjointSet` vs `num::DisjointSet32`, `num::IndexedPriorityQueue` addressable heaps, graph Laplacian assembly | Standard algorithms & memory footprint |
