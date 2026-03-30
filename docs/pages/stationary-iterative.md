# Stationary Iterative Solvers: Jacobi and Gauss-Seidel {#page_stationary_notes}

## Problem Statement

Solve \f$Ax = b\f$ where \f$A \in \mathbb{R}^{n \times n}\f$ is square and non-singular. Stationary iterative methods split \f$A = M - N\f$ and iterate:

\f[Mx_{k+1} = Nx_k + b \quad \Longrightarrow \quad x_{k+1} = \underbrace{M^{-1}N}_{B} x_k + M^{-1}b\f]

Convergence requires the spectral radius \f$\rho(B) < 1\f$.

Decompose \f$A = D + L + U\f$ (diagonal, strictly lower, strictly upper triangular):

| Method | \f$M\f$ | \f$N\f$ | Update uses |
|--------|-----|-----|-------------|
| Jacobi | \f$D\f$ | \f$-(L+U)\f$ | old \f$x\f$ only |
| Gauss-Seidel | \f$D+L\f$ | \f$-U\f$ | mixed old/new \f$x\f$ |
| SOR(\f$\omega\f$) | \f$\frac{1}{\omega}D + L\f$ | \f$(\frac{1}{\omega}-1)D - U\f$ | mixed + relaxation |

---

## Part I -- Jacobi Iteration

### Algorithm

Each component of the new iterate is computed **independently** from the old one:

\f[x_i^{(k+1)} = \frac{1}{A_{ii}}\!\left(b_i - \sum_{j \neq i} A_{ij}\, x_j^{(k)}\right), \quad i = 0, 1, \ldots, n-1\f]

```
function Jacobi(A, b, x, tol, max_iter):
    for iter = 1 to max_iter:
        for i = 0 to n-1:
            x_new[i] <- (b[i] - sum_{j != i} A[i,j]*x[j]) / A[i,i]
        x <- x_new                            // simultaneous update

        if ||b - Ax||_2 < tol: return x, converged
```

**Key property**: all new values \f$x_i^{(k+1)}\f$ are computed from the **same** old iterate \f$x^{(k)}\f$, making Jacobi **embarrassingly parallel**.

**Flops per iteration**: \f$2n^2\f$ (equivalent to one full matvec).

### Convergence Conditions

Jacobi converges if any of the following hold:
1. **Strict diagonal dominance**: \f$|A_{ii}| > \sum_{j \neq i} |A_{ij}|\f$ for all \f$i\f$
2. **Symmetric positive definiteness** (sufficient but not necessary)

The iteration matrix is \f$B_J = -D^{-1}(L+U)\f$, and the convergence factor per iteration is \f$\rho(B_J)\f$.

For the model Poisson problem (5-point stencil on an \f$n \times n\f$ grid):

\f[\rho(B_J) = \cos\!\left(\frac{\pi}{n+1}\right) \approx 1 - \frac{\pi^2}{2n^2} \quad (n \to \infty)\f]

This requires \f$O(n^2)\f$ iterations to reduce the error by \f$1/e\f$ -- extremely slow for large \f$n\f$.

### Jacobi as a Multigrid Smoother

Though useless as a standalone solver for large problems, Jacobi is an effective **smoother** in multigrid: 1-2 Jacobi sweeps efficiently damp high-frequency error components (Fourier modes with wavenumber \f$k > n/2\f$). Low-frequency modes are handled by coarser grids.

---

## Part II -- Gauss-Seidel Iteration

### Algorithm

Each component uses the **most recently computed** values -- \f$x_j^{(k+1)}\f$ for \f$j < i\f$, \f$x_j^{(k)}\f$ for \f$j > i\f$:

\f[x_i^{(k+1)} = \frac{1}{A_{ii}}\!\left(b_i - \sum_{j < i} A_{ij}\, x_j^{(k+1)} - \sum_{j > i} A_{ij}\, x_j^{(k)}\right)\f]

```
function Gauss_Seidel(A, b, x, tol, max_iter):
    for iter = 1 to max_iter:
        for i = 0 to n-1:
            sigma <- sum_{j != i} A[i,j] * x[j]    // uses latest values
            x[i]  <- (b[i] - sigma) / A[i,i]        // immediate in-place update

        if ||b - Ax||_2 < tol: return x, converged
```

**Flops per iteration**: \f$2n^2\f$ (same as Jacobi).

### Gauss-Seidel vs Jacobi

For SPD \f$A\f$, Gauss-Seidel always converges and satisfies the Stein-Rosenberg relation:

\f[\rho(B_{GS}) = \rho(B_J)^2\f]

For the model Poisson problem:

\f[\rho(B_{GS}) = \cos^2\!\left(\frac{\pi}{n+1}\right) \approx 1 - \frac{\pi^2}{n^2}\f]

GS converges in **half as many iterations** as Jacobi: the fresh data incorporated immediately does more work per step.

### SOR -- Successive Over-Relaxation

The GS update is a correction \f$\Delta x_i = \tilde{x}_i - x_i\f$. Over-relax by factor \f$\omega \in (0, 2)\f$:

\f[x_i^{(k+1)} = x_i^{(k)} + \omega\bigl(\tilde{x}_i^{(GS)} - x_i^{(k)}\bigr)\f]

For the model Poisson problem, the **optimal relaxation parameter** is:

\f[\omega_{\text{opt}} = \frac{2}{1 + \sin(\pi/(n+1))} \approx 2 - \frac{2\pi}{n}\f]

which gives:

\f[\rho(B_{SOR,\omega_{\text{opt}}}) = (\omega_{\text{opt}} - 1)^2 \approx \left(1 - \frac{2\pi}{n}\right)^2 \approx 1 - \frac{4\pi}{n}\f]

SOR with optimal \f$\omega\f$ requires only \f$O(n)\f$ iterations -- a dramatic improvement over \f$O(n^2)\f$ for GS. This was a landmark achievement of 1950s-60s computational science (D.M. Young, 1950).

---

## Ordering and Parallelism

Sequential GS (natural ordering \f$i = 0, 1, \ldots, n-1\f$) has a data dependency chain: row \f$i\f$ depends on rows \f$0, \ldots, i-1\f$ being updated. This is **inherently sequential**.

### Red-Black Ordering

For the 5-point stencil on a structured grid, color node \f$(i,j)\f$ red if \f$i+j\f$ is even, black otherwise. Red nodes depend only on black neighbors:

\f[x_{ij}^{(\text{new})} = \frac{1}{4}\bigl(x_{i-1,j} + x_{i+1,j} + x_{i,j-1} + x_{i,j+1} - h^2 f_{ij}\bigr)\f]

All red updates are independent -> one parallel sweep. All black updates are independent -> another parallel sweep.

```
// Red-black GS: 2 parallel sweeps per iteration
parallel for all (i,j) with i+j even: update x[i,j]
parallel for all (i,j) with i+j odd:  update x[i,j]
```

Red-black GS has the same convergence rate as standard GS on Poisson-type problems but is fully parallelizable. Widely used in GPU stencil solvers.

### Multi-Color Ordering

For unstructured meshes, find a proper coloring of the adjacency graph of \f$A\f$: nodes of the same color have no edges (i.e., the corresponding off-diagonal entries of \f$A\f$ are zero). All nodes of the same color can be updated in parallel. Minimum colors = chromatic number (typically 5-10 for 3D unstructured FEM meshes).

---

## Performance Optimization

### SIMD for the Row Dot Product

The inner loop `sigma += A[i,j] * x[j]` is a dense row dot product:

```cpp
__m256d acc = _mm256_setzero_pd();
for (idx j = 0; j < n; j += 4) {
    __m256d va = _mm256_load_pd(&A(i, j));
    __m256d vx = _mm256_load_pd(&x[j]);
    acc = _mm256_fmadd_pd(va, vx, acc);
}
real sigma = hsum(acc) - A(i,i) * x[i];   // subtract diagonal contribution
```

FMA throughput: 8 flops/cycle on AVX2. For large \f$n\f$, this is bandwidth-limited.

### Cache Blocking for Large n

For \f$n\f$ large enough that the matrix does not fit in L3:

```
// Tiled Jacobi: process rows in blocks of B to reuse x in cache
for ib = 0, B, 2B, ...:
    for i = ib to min(ib+B, n)-1:
        sigma[i] = row_dot(A, row i, x)   // x reused B times per cache load
```

Choose \f$B\f$ so that \f$B \times n \times 8\,\text{bytes} \lesssim L_2\f$. For \f$n = 4096\f$ and \f$L_2 = 512\,\text{KB}\f$: \f$B = 512\,\text{KB}/(4096 \times 8) = 16\f$.

### Sparse Gauss-Seidel (CSR Format)

For sparse \f$A\f$ stored in CSR format, the GS update is:

```cpp
for (idx i = 0; i < n; ++i) {
    real sigma = b[i];
    for (idx ptr = row_ptr[i]; ptr < row_ptr[i+1]; ++ptr)
        if (col_idx[ptr] != i)
            sigma -= val[ptr] * x[col_idx[ptr]];    // irregular access
    x[i] = sigma / diag[i];
}
```

The irregular access `x[col_idx[ptr]]` is cache-unfriendly for unstructured meshes. Reordering by **Reverse Cuthill-McKee (RCM)** or **AMD (Approximate Minimum Degree)** reduces matrix bandwidth and improves cache reuse.

### Parallelism Summary

| Method | Parallel structure | Comment |
|--------|-------------------|---------|
| Jacobi | All \f$i\f$ independent | Trivial OpenMP / CUDA |
| GS (natural ordering) | Sequential chain | Cannot parallelize directly |
| GS (red-black) | 2 parallel phases | Structured grids |
| GS (multi-color) | \f$c\f$ parallel phases | \f$c\f$ = chromatic number |
| SOR | Same as GS base ordering | With relaxation \f$\omega\f$ |

For modern hardware, **preconditioned CG or GMRES** almost always outperforms Gauss-Seidel as a standalone solver. GS remains valuable as a multigrid smoother, a simple preconditioner (symmetric GS = SSOR), and an educational baseline.
