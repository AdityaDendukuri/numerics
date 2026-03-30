# One-Sided Jacobi SVD {#page_svd_jacobi_notes}

## Problem Statement

Given \f$A \in \mathbb{R}^{m \times n}\f$ with \f$m \geq n\f$, find the singular value decomposition:

\f[A = U \Sigma V^T, \qquad U \in \mathbb{R}^{m \times r},\ \Sigma = \operatorname{diag}(\sigma_1, \ldots, \sigma_r),\ V \in \mathbb{R}^{n \times r}\f]

where \f$r = \min(m, n)\f$, \f$U^TU = I\f$, \f$V^TV = I\f$, and \f$\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r \geq 0\f$ are the **singular values**.

The one-sided Jacobi method applies Givens rotations to the **columns** of \f$A\f$ from the right until they are mutually orthogonal. Column norms then give singular values; normalized columns give left singular vectors.

---

## Mathematical Foundation

### Relation to the Eigenvalue Problem

The singular values of \f$A\f$ are the square roots of the eigenvalues of \f$A^TA\f$ (or \f$AA^T\f$):

\f[A^TA = V \Sigma^2 V^T\f]

Zeroing \f$[A^TA]_{pq}\f$ is equivalent to orthogonalizing columns \f$p\f$ and \f$q\f$ of \f$A\f$. The one-sided method works directly on \f$A\f$, never forming \f$A^TA\f$ explicitly (which would square the condition number).

### The Column Orthogonality Condition

Define the inner products of columns \f$p\f$ and \f$q\f$:

\f[\alpha = \|a_p\|^2 = \sum_i A_{ip}^2, \qquad \beta = \|a_q\|^2 = \sum_i A_{iq}^2, \qquad \gamma = a_p \cdot a_q = \sum_i A_{ip} A_{iq}\f]

The **relative cosine** \f$|\gamma|/\sqrt{\alpha\beta} = |\cos\angle(a_p, a_q)|\f$ measures deviation from orthogonality. Rotation is skipped when \f$|\gamma|/\sqrt{\alpha\beta} < \varepsilon_{\text{tol}}\f$.

### Rotation Angle to Zero \f$[A^TA]_{pq}\f$

The right rotation \f$A \leftarrow AG\f$ with \f$G\f$ a Givens matrix in the \f$(p,q)\f$ plane zeros \f$[A^TA]_{pq} = \gamma\f$ when (setting the new inner product to zero):

\f[\tan(2\theta) = \frac{2\gamma}{\beta - \alpha} \qquad \Longrightarrow \qquad \zeta = \frac{\beta - \alpha}{2\gamma}, \quad t = \frac{\operatorname{sign}(\zeta)}{|\zeta| + \sqrt{1 + \zeta^2}}\f]

\f[c = \frac{1}{\sqrt{1+t^2}}, \qquad s = ct\f]

This formula uses the **smaller root** of \f$t^2 + 2\zeta t - 1 = 0\f$ -- consistent with \f$|t| \leq 1\f$, i.e., \f$|\theta| \leq \pi/4\f$.

**Sign verification**: with rotation \f$a_p' = c\, a_p - s\, a_q\f$, \f$a_q' = s\, a_p + c\, a_q\f$, the new inner product is:

\f[a_p'^T a_q' = cs(\alpha - \beta) + (c^2 - s^2)\gamma = 0 \quad \checkmark\f]

(this is \f$c^2 - s^2 = 1 - 2t^2/(1+t^2)\f$ and \f$2cs = 2t/(1+t^2)\f$, so \f$(c^2-s^2)\gamma + 2cs \cdot (\alpha-\beta)/2 = 0\f$ by construction of \f$t\f$).

---

## Algorithm

```
function svd(A, tol, max_sweeps):
    m, n <- A.rows, A.cols
    r    <- min(m, n)
    V    <- I_n                      // accumulates right singular vectors

    for sweep = 1 to max_sweeps:
        max_cos <- 0

        for p = 0 to r-2:
            for q = p+1 to r-1:
                // Inner products of columns p and q
                alpha <- sum_i A[i,p]^2
                beta  <- sum_i A[i,q]^2
                gamma <- sum_i A[i,p] * A[i,q]

                if alpha < 1e-300 or beta < 1e-300: continue

                cos_pq <- |gamma| / sqrt(alpha * beta)
                max_cos <- max(max_cos, cos_pq)
                if cos_pq < tol: continue             // already orthogonal enough

                // Rotation parameters
                zeta <- (beta - alpha) / (2 * gamma)
                t    <- sign(zeta) / (|zeta| + sqrt(1 + zeta^2))
                c    <- 1 / sqrt(1 + t^2);  s <- c * t

                // Update columns of A
                for i = 0..m-1:
                    (A[i,p], A[i,q]) <- (c*A[i,p] - s*A[i,q],  s*A[i,p] + c*A[i,q])

                // Accumulate right singular vectors
                for i = 0..n-1:
                    (V[i,p], V[i,q]) <- (c*V[i,p] - s*V[i,q],  s*V[i,p] + c*V[i,q])

        if max_cos < tol: break     // all column pairs orthogonal enough

    // Extract singular values and left singular vectors
    for j = 0..r-1:
        sigma[j] <- ||A[:,j]||
        U[:,j]   <- A[:,j] / sigma[j]   (if sigma[j] > 0)

    // Sort descending by singular value; permute U, V columns to match
    selection_sort(sigma, U, V)

    // Return economy Vt (r x n)
    Vt[i,j] <- V[j,i]   for i < r

    return (U, sigma, Vt, sweeps, converged)
```

**Complexity**: \f$O(r^2)\f$ rotations per sweep \f$\times\f$ \f$O(m + n)\f$ flops per rotation = \f$O(r^2(m+n))\f$ per sweep.

**Convergence**: typically 5-10 sweeps for double precision, similar to the symmetric Jacobi eigensolver.

---

## Convergence Analysis

Define the **off-orthogonality** of \f$A\f$:

\f[\psi(A) = \sum_{p < q} \bigl([A^TA]_{pq}\bigr)^2 = \sum_{p < q} \gamma_{pq}^2\f]

Each rotation decreases \f$\psi\f$ by \f$\gamma_{pq}^2\f$:

\f[\psi(A') = \psi(A) - \gamma_{pq}^2\f]

After a full sweep visiting all \f$\binom{r}{2}\f$ pairs:

\f[\psi(A_{\text{sweep}+1}) \leq \left(1 - \frac{2}{r(r-1)}\right) \psi(A_{\text{sweep}}) \qquad \text{(linear phase)}\f]

Ultimate quadratic convergence: once \f$\max_{p \neq q} |[A^TA]_{pq}|\f$ is small, convergence transitions to a quadratic phase -- same analysis as symmetric Jacobi applied to \f$A^TA\f$.

---

## Accuracy Properties

One-sided Jacobi SVD is among the most accurate SVD algorithms:

1. **High relative accuracy for small singular values**: unlike bidiagonalization + QR, the Jacobi SVD can compute small singular values to high relative accuracy even when \f$\kappa(A) \gg 1/\varepsilon_{\text{mach}}\f$, provided the columns of \f$A\f$ are well-scaled. Formally, if \f$\sigma_k / \sigma_1 = O(\varepsilon_{\text{mach}})\f$, Jacobi still computes \f$\sigma_k\f$ with relative error \f$O(\varepsilon_{\text{mach}})\f$.

2. **Column-scaled accuracy**: the relative accuracy of \f$\sigma_k\f$ is \f$O(\varepsilon_{\text{mach}} \cdot \kappa(D^{-1}A))\f$ where \f$D = \operatorname{diag}(\|a_1\|, \ldots, \|a_n\|)\f$ is the column scaling. If \f$A\f$ is column-equilibrated (\f$D \approx I\f$), all singular values are computed to full relative precision.

This contrasts with standard bidiagonalization (LAPACK `dgesdd`), which computes all singular values to absolute accuracy \f$O(\varepsilon_{\text{mach}} \|A\|)\f$ -- relative accuracy only for \f$\sigma_k \gtrsim \varepsilon_{\text{mach}} \|A\|\f$.

---

## Comparison with Alternatives

| Algorithm | Cost | Accuracy | Use when |
|-----------|------|----------|----------|
| One-sided Jacobi (ours) | \f$O(r^2(m+n) \times \text{sweeps})\f$ | High relative | Small \f$n\f$, need small \f$\sigma_k\f$ accurately |
| Bidiagonalization + QR (LAPACK `dgesvd`) | \f$O(mn^2)\f$ | Absolute | Standard dense SVD |
| Divide-and-conquer (LAPACK `dgesdd`) | \f$O(mn^2)\f$ | Absolute | Fast, most common |
| Randomized SVD | \f$O(mn k / \varepsilon)\f$ | Approximate | \f$k \ll \min(m,n)\f$ |
| Golub-Reinsch | \f$O(mn^2)\f$ | Absolute | Historical reference |

For \f$m \gg n\f$ (tall-skinny): first compute thin QR (\f$A = \hat{Q}\hat{R}\f$), then Jacobi SVD on \f$\hat{R} \in \mathbb{R}^{n \times n}\f$. Left singular vectors: \f$U = \hat{Q}\hat{U}\f$.

---

## Performance Optimization

### Current Implementation

Two nested loops over column pairs \f$(p, q)\f$, each requiring:
- Three inner products (\f$\alpha\f$, \f$\beta\f$, \f$\gamma\f$): \f$3m\f$ FMAs -> bandwidth limited
- Column rotation of \f$A\f$: \f$4m\f$ FMAs
- Column rotation of \f$V\f$: \f$4n\f$ FMAs

Total per rotation: \f$(3 + 4)m + 4n \approx 7m\f$ flops for \f$m \gg n\f$.

### SIMD for Column Inner Products and Rotation

Fuse the three inner product computations and the rotation in one pass:

```cpp
// Step 1: compute alpha, beta, gamma (three accumulators, one pass over columns p,q)
__m256d va = _mm256_setzero_pd();   // alpha
__m256d vb = _mm256_setzero_pd();   // beta
__m256d vg = _mm256_setzero_pd();   // gamma
for (idx i = 0; i < m; i += 4) {
    __m256d ap = _mm256_load_pd(&A(i, p));
    __m256d aq = _mm256_load_pd(&A(i, q));
    va = _mm256_fmadd_pd(ap, ap, va);
    vb = _mm256_fmadd_pd(aq, aq, vb);
    vg = _mm256_fmadd_pd(ap, aq, vg);
}
alpha = hsum(va);  beta = hsum(vb);  gamma = hsum(vg);

// Step 2: rotation (second pass)
__m256d vc  = _mm256_broadcast_sd(&c);
__m256d vs  = _mm256_broadcast_sd(&s);
for (idx i = 0; i < m; i += 4) {
    __m256d ap = _mm256_load_pd(&A(i, p));
    __m256d aq = _mm256_load_pd(&A(i, q));
    _mm256_store_pd(&A(i,p), _mm256_fmsub_pd(vc,ap, _mm256_mul_pd(vs,aq)));
    _mm256_store_pd(&A(i,q), _mm256_fmadd_pd(vs,ap, _mm256_mul_pd(vc,aq)));
}
```

Two passes per rotation pair (one for inner products, one for the update). The data \f$A[:,p]\f$ and \f$A[:,q]\f$ are loaded twice; for column-major storage they are contiguous and fit in L1 for small \f$m\f$.

### Fused Single-Pass Update

When \f$m \leq L_1 / (2 \times 8\,\text{bytes})\f$ (both columns fit in L1, i.e., \f$m \leq 4096\f$ for 64 KB L1):

```cpp
// Read-compute-write in one pass: inner products AND rotation together
// Requires updating V with the same (c,s) computed on-the-fly -- NOT POSSIBLE
// (c,s depend on gamma which requires a full pass first)
```

Cannot fuse into one pass because \f$c, s\f$ depend on \f$\gamma\f$ which requires a full scan. For large \f$m\f$ (columns don't fit in cache), the two-pass approach causes 2 column pair reads from DRAM -- unavoidable unless the rotation is batched.

### Batched Rotations (Row-Blocked Update)

Process all \f$\binom{r}{2}\f$ rotations in a sweep as a sequence of BLAS-2 DROT operations, then bulk-apply them to \f$V\f$ using DGEMM:

```
// Phase 1: Compute all (zeta, t, c, s) for the sweep
for all (p,q) pairs: compute (c_{pq}, s_{pq}) from A

// Phase 2: Apply rotations to A column pairs (BLAS-1 DROT x r^2/2)
for all (p,q) pairs: drot(m, A[:,p], A[:,q], c_pq, s_pq)

// Phase 3: Accumulate V as product of all Givens matrices in DGEMM form
// V_new = V_old * G_1 * G_2 * ... (product of Givens -> QR of random matrix -> compact)
```

For the \f$V\f$ accumulation, represent the sequence of \f$\binom{r}{2}\f$ Givens rotations as a single orthogonal matrix via their WY product representation, then apply with one DGEMM of size \f$n \times n \times n\f$ -- BLAS-3 instead of \f$O(r^2)\f$ BLAS-1 DROT calls.

### Pre-Processing: Column Equilibration

Before running Jacobi, normalize columns:

\f[A \leftarrow A D^{-1}, \quad D = \operatorname{diag}(\|a_1\|, \ldots, \|a_n\|)\f]

After convergence, \f$\sigma_j^{\text{actual}} = \sigma_j^{\text{computed}} \times D_{jj}\f$.

Column equilibration improves convergence rate (all column pairs start at roughly equal norm -> more uniform convergence) and is essential for the high relative accuracy property.

### Block One-Sided Jacobi

Instead of scalar rotations, process \f$b \times b\f$ blocks of column pairs simultaneously:

1. Extract the \f$2b \times 2b\f$ block \f$B = \begin{pmatrix} A[:,P]^T A[:,P] & A[:,P]^T A[:,Q] \\ A[:,Q]^T A[:,P] & A[:,Q]^T A[:,Q] \end{pmatrix}\f$
2. Compute orthogonal \f$G \in \mathbb{R}^{2b \times 2b}\f$ that diagonalizes \f$B\f$ via small \f$2b \times 2b\f$ Jacobi
3. Update \f$[A[:,P], A[:,Q]] \leftarrow [A[:,P], A[:,Q]] \cdot G\f$ via DGEMM

The dominant cost is the DGEMM: \f$O(m \cdot 2b \cdot 2b) = O(4mb^2)\f$ with arithmetic intensity \f$O(b)\f$. This is the **block Jacobi SVD** -- achieves near-peak GFLOP/s for the column update phase.
