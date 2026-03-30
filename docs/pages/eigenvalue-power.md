# Power Iteration, Inverse Iteration, and Rayleigh Quotient Iteration {#page_eig_power_notes}

## Background

All three methods find **one** eigenvalue/eigenvector pair at a time by repeatedly applying a matrix (or its inverse) to a vector. They trade off cost per iteration against convergence rate:

| Method | Cost/iteration | Convergence | Finds |
|--------|---------------|-------------|-------|
| Power iteration | 1 matvec \f$O(n^2)\f$ | Linear, rate \f$|\lambda_1/\lambda_2|\f$ | Dominant eigenvalue |
| Inverse iteration | 1 triangular solve \f$O(n^2)\f$ | Linear, rate \f$|(\lambda-\sigma)/(\lambda_2-\sigma)|\f$ | Eigenvalue nearest \f$\sigma\f$ |
| Rayleigh quotient | 1 LU factorization \f$O(n^3)\f$ | Cubic | Eigenvalue nearest starting guess |

---

## Part I -- Power Iteration

### Algorithm

Expand the starting vector in the eigenbasis of \f$A\f$: \f$v_0 = \sum_i \alpha_i u_i\f$. After \f$k\f$ multiplications:

\f[A^k v_0 = \lambda_1^k \!\left(\alpha_1 u_1 + \sum_{i \geq 2} \alpha_i \left(\frac{\lambda_i}{\lambda_1}\right)^k u_i\right)\f]

The component along \f$u_i\f$ (\f$i \geq 2\f$) decays as \f$(\lambda_i/\lambda_1)^k\f$ -> converges to \f$u_1\f$.

```
function power_iteration(A, tol, max_iter):
    v <- ones(n) / ||ones(n)||       // random start also valid

    for iter = 1 to max_iter:
        w      <- A v                // matvec: dominant cost O(n^2)
        lambda <- v^T w              // Rayleigh quotient (v is unit)
        v      <- w / ||w||         // renormalise

        if |lambda - lambda_prev| < tol: return lambda, v, converged
```

**Flops per iteration**: \f$2n^2\f$ (matvec) + \f$2n\f$ (Rayleigh quotient) + \f$n\f$ (normalise).

### Convergence Analysis

Let \f$|\lambda_1| > |\lambda_2| \geq \cdots \geq |\lambda_n|\f$. The asymptotic convergence factor is:

\f[r = \left|\frac{\lambda_2}{\lambda_1}\right|\f]

The angle between \f$v_k\f$ and \f$u_1\f$ satisfies:

\f[\tan \angle(v_k, u_1) \leq \left|\frac{\lambda_2}{\lambda_1}\right|^k \frac{\|\alpha_{2:n}\|}{|\alpha_1|}\f]

If \f$\lambda_1 \gg \lambda_2\f$ (e.g., Google PageRank with \f$r \approx 0.85\f$): fast convergence. If \f$|\lambda_1| \approx |\lambda_2|\f$: very slow -- use Lanczos instead.

### Failure Modes

1. **\f$\alpha_1 = 0\f$**: initial vector orthogonal to \f$u_1\f$ -> converges to \f$\lambda_2\f$ instead. Floating-point roundoff usually introduces a component along \f$u_1\f$, but use a random start to be safe.
2. **\f$|\lambda_1| = |\lambda_2|\f$**: complex conjugate pair or \f$\pm\f$ pair -- iterate oscillates. Use 2-vector subspace iteration.
3. **Slow convergence**: if \f$r = 0.99\f$, \f$\log(10)/\log(1/0.99) \approx 229\f$ iterations per digit.

### Deflation (Hotelling)

After finding \f$(\lambda_1, u_1)\f$, subtract to expose \f$\lambda_2\f$:

\f[A' = A - \lambda_1 u_1 u_1^T \quad \Rightarrow \quad \text{eigenvalues of } A': \lambda_2, \lambda_3, \ldots, \lambda_n, 0\f]

Errors accumulate with each deflation -- numerically inferior to Lanczos or QR iteration for finding many eigenvalues.

---

## Part II -- Inverse Iteration

### Algorithm

Replace \f$A\f$ with \f$(A - \sigma I)^{-1}\f$. Its eigenvalues are \f$\{1/(\lambda_i - \sigma)\}\f$, so the dominant eigenvalue of \f$(A-\sigma I)^{-1}\f$ corresponds to the \f$\lambda_i\f$ **closest to** \f$\sigma\f$:

```
function inverse_iteration(A, sigma, tol, max_iter):
    // Factor (A - sigma*I) once: O(n^3)
    LU <- lu_factor(A - sigma*I)

    v <- ones(n) / ||ones(n)||

    for iter = 1 to max_iter:
        w      <- lu_solve(LU, v)   // O(n^2) triangular solve
        v      <- w / ||w||
        lambda <- v^T A v            // Rayleigh quotient

        if |lambda - lambda_prev| < tol: return lambda, v, converged
```

**Key advantage**: factorize once (O(n^3)), then solve cheaply (O(n^2)) at each iteration.

### Convergence

The convergence factor at shift \f$\sigma\f$ is:

\f[r = \frac{|\lambda - \sigma|}{|\lambda_2 - \sigma|}\f]

where \f$\lambda\f$ is the closest eigenvalue and \f$\lambda_2\f$ is the second closest. Choosing \f$\sigma\f$ close to the target makes \f$r \approx 0\f$: typically 3-5 iterations suffice for machine precision once \f$\sigma\f$ is within 10% of \f$\lambda\f$.

**Stability paradox**: when \f$\sigma \approx \lambda\f$ exactly, \f$(A - \sigma I)\f$ is nearly singular -- the linear system is ill-conditioned. Yet the solution **direction** is extremely well-determined (the error in the eigenvalue is exponentially small). The most ill-conditioned solve gives the most accurate eigenvector.

### Use Cases

1. **Refine a rough eigenvalue estimate**: one inverse iteration step from within 10% of \f$\lambda\f$ gives machine-precision accuracy
2. **Interior eigenvalue**: choose \f$\sigma\f$ in the middle of the spectrum -- direct power iteration cannot reach this
3. **Sturm sequence + bisection**: bisect on \f$[a,b]\f$ to isolate \f$\lambda\f$, then inverse-iterate at \f$\sigma = (a+b)/2\f$

---

## Part III -- Rayleigh Quotient Iteration (RQI)

### Algorithm

Update the shift \f$\sigma\f$ at every iteration using the Rayleigh quotient of the current vector \f$v\f$:

\f[\sigma_k = \frac{v_k^T A v_k}{v_k^T v_k} = v_k^T A v_k \quad (\text{since } \|v_k\| = 1)\f]

Fresh factorization at each step is expensive but gives **cubic convergence**:

```
function rayleigh_iteration(A, x0, tol, max_iter):
    v     <- x0 / ||x0||
    sigma <- v^T A v

    for iter = 1 to max_iter:
        LU <- lu_factor(A - sigma*I)     // O(n^3) per iter!
        if singular: return sigma, v     // sigma is exact eigenvalue

        w     <- lu_solve(LU, v)
        v     <- w / ||w||
        sigma <- v^T A v

        res <- ||A v - sigma v||_2
        if res < tol: return sigma, v, converged
```

**Cost per iteration**: \f$O(n^3)\f$ -- same as a direct solve. Practical only for small \f$n\f$ or as a refinement step (3 iterations at most from a good starting guess).

### Convergence: Cubic Rate

For symmetric \f$A\f$, the Rayleigh quotient is a second-order approximation to the eigenvalue:

\f[\sigma = \lambda + O(\|v - u\|^2)\f]

(the error in \f$\sigma\f$ is quadratic in the eigenvector error). The new eigenvector error is proportional to \f$|\sigma - \lambda|\f$ -> quadratic in the old eigenvector error -> **cubic overall**:

\f[|\sigma_{k+1} - \lambda| \leq C\, |\sigma_k - \lambda|^3\f]

**Consequence**: if the current estimate has error \f$10^{-4}\f$, the next has error \f$\sim 10^{-12}\f$ -- machine precision in one more step.

For non-symmetric \f$A\f$, only **quadratic** convergence (the second-order property of the Rayleigh quotient holds for symmetric matrices only).

### Practical Strategy: Combine Methods

```
Phase 1: Lanczos (k steps)       -> Ritz values lambda~_1, ..., lambda~_k   O(k*nnz) or O(kn^2)
Phase 2: One inverse iteration step at sigma = lambda~i               O(n^3) x few steps
```

Or for dense \f$A\f$:
```
Phase 1: power_iteration or inverse_iteration -> rough (lambda, v)
Phase 2: one more inverse_iteration at sigma = lambda  -> machine precision
```

---

## Performance Optimization

### SIMD for Power Iteration Matvec

Power iteration's only per-iteration cost is the matvec \f$w \leftarrow Av\f$. For row-major dense \f$A\f$:

```cpp
for (idx i = 0; i < n; ++i) {
    __m256d acc = _mm256_setzero_pd();
    for (idx j = 0; j < n; j += 4)
        acc = _mm256_fmadd_pd(_mm256_load_pd(&A(i,j)),
                               _mm256_load_pd(&v[j]), acc);
    w[i] = hsum(acc);
}
```

For sparse \f$A\f$: use library SpMV (MKL `mkl_dcsrmv`, cuSPARSE `cusparseDbsrmv`).

### Subspace Iteration (Block Power Method)

Instead of one vector, maintain \f$V \in \mathbb{R}^{n \times k}\f$ and apply \f$A\f$:

\f[W \leftarrow AV \quad (\text{DGEMM: } O(n^2 k)), \qquad [Q, R] \leftarrow \text{thin QR}(W), \qquad V \leftarrow Q\f]

Simultaneously finds the \f$k\f$ dominant eigenpairs. Convergence factors:

\f[r_i = \left|\frac{\lambda_{k+1}}{\lambda_i}\right|, \quad i = 1, \ldots, k\f]

The blocked version amortizes matvec overhead (DGEMM \f$\gg\f$ GEMV in practice) and is the precursor to the Lanczos algorithm.

### Blocked Inverse Iteration for Multiple Eigenvectors

For tridiagonal \f$T\f$ (from Lanczos), inverse-iterate all desired eigenvectors simultaneously:

\f[\text{Solve } (T - \sigma I)\,X = B \quad \text{(all RHS simultaneously)}\f]

Replace \f$k\f$ DTRSV calls with one DTRSM call (BLAS-3) -> much higher arithmetic intensity.

### Shift Selection via Gershgorin Discs

Eigenvalues of \f$A\f$ lie in the union of Gershgorin discs:

\f[\sigma(A) \subseteq \bigcup_{i=1}^n \bigl\{ z \in \mathbb{C} : |z - A_{ii}| \leq R_i \bigr\}, \qquad R_i = \sum_{j \neq i} |A_{ij}|\f]

Use Gershgorin to get rough bounds for initial shift selection in inverse iteration. For SPD matrices, all discs lie on the positive real axis: \f$[A_{ii} - R_i, A_{ii} + R_i] \cap \mathbb{R}_{>0}\f$.

**Sturm sequence bisection** for SPD tridiagonal \f$T\f$: the number of eigenvalues of \f$T\f$ less than \f$\mu\f$ equals the number of sign changes in the sequence \f$\{d_k\}\f$ defined by:

\f[d_0 = T_{00} - \mu, \qquad d_k = (T_{kk} - \mu) - \frac{T_{k-1,k}^2}{d_{k-1}}\f]

Bisect on \f$[a, b]\f$ to isolate each eigenvalue to within \f$\varepsilon\f$ in \f$O(n \log(1/\varepsilon))\f$ flops.
