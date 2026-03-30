# Symmetric Jacobi Eigendecomposition (eig_sym) {#page_eig_jacobi_notes}

## Problem Statement

Given \f$A \in \mathbb{R}^{n \times n}\f$ **symmetric** (\f$A = A^T\f$), find the spectral decomposition:

\f[A = V \Lambda V^T, \qquad V^TV = I, \quad \Lambda = \operatorname{diag}(\lambda_1, \ldots, \lambda_n)\f]

The Jacobi method is an iterative similarity transform that drives all off-diagonal entries to zero by repeatedly applying **Givens (plane) rotations**.

---

## Mathematical Foundation

### Givens Rotation in the \f$(p,q)\f$ Plane

The Givens rotation \f$G(p, q, \theta)\f$ is the identity with a \f$2 \times 2\f$ rotation block at rows/columns \f$p\f$ and \f$q\f$:

\f[G_{pp} = c,\quad G_{pq} = -s,\quad G_{qp} = s,\quad G_{qq} = c, \qquad c = \cos\theta,\ s = \sin\theta\f]

The similarity transform \f$A' = G^T A G\f$ modifies only rows/columns \f$p\f$ and \f$q\f$:

\f[A'_{pp} = c^2 A_{pp} - 2cs\, A_{pq} + s^2 A_{qq}\f]
\f[A'_{qq} = s^2 A_{pp} + 2cs\, A_{pq} + c^2 A_{qq}\f]
\f[A'_{pq} = A'_{qp} = (c^2 - s^2) A_{pq} + cs(A_{pp} - A_{qq}) \quad \leftarrow \text{want } = 0\f]
\f[A'_{rp} = A'_{pr} = c\, A_{rp} - s\, A_{rq}, \quad r \neq p, q\f]
\f[A'_{rq} = A'_{qr} = s\, A_{rp} + c\, A_{rq}, \quad r \neq p, q\f]

### Choosing \f$\theta\f$ to Zero \f$A'_{pq}\f$

Set \f$A'_{pq} = 0\f$:

\f[(c^2 - s^2) A_{pq} + cs(A_{pp} - A_{qq}) = 0\f]

Let \f$t = \tan\theta\f$ and \f$\tau = (A_{qq} - A_{pp})/(2A_{pq})\f$:

\f[t^2 + 2\tau t - 1 = 0 \quad \Longrightarrow \quad t = \frac{\operatorname{sign}(\tau)}{|\tau| + \sqrt{1 + \tau^2}}\f]

Taking the **smaller root** \f$|t| \leq 1\f$ (i.e., \f$|\theta| \leq \pi/4\f$) minimizes the perturbation to other off-diagonal entries. Then:

\f[c = \frac{1}{\sqrt{1 + t^2}}, \qquad s = ct\f]

The formula avoids catastrophic cancellation when \f$\tau \gg 1\f$: in that case \f$t \approx 1/(2\tau) \approx 0\f$, so \f$c \approx 1\f$ and \f$s \approx 0\f$ -- almost no rotation is needed.

---

## Algorithm: Cyclic Jacobi

Visit every off-diagonal pair \f$(p, q)\f$ in a fixed cyclic order (row-by-row sweep). The off-diagonal Frobenius norm \f$\sigma_{\text{off}} = \sqrt{2\sum_{p<q} A_{pq}^2}\f$ is the convergence monitor:

```
function eig_sym(A, tol, max_sweeps):
    n <- A.rows
    V <- I_n

    for sweep = 1 to max_sweeps:
        off <- sqrt(2 * sum_{p<q} A[p,q]^2)
        if off < tol: break

        for p = 0 to n-2:
            for q = p+1 to n-1:
                if |A[p,q]| < 1e-15: continue

                tau <- (A[q,q] - A[p,p]) / (2 * A[p,q])
                t   <- sign(tau) / (|tau| + sqrt(1 + tau^2))
                c   <- 1 / sqrt(1 + t^2);  s <- c * t

                // Update diagonal and zero (p,q)
                A'[p,p] <- c^2*A[p,p] - 2cs*A[p,q] + s^2*A[q,q]
                A'[q,q] <- s^2*A[p,p] + 2cs*A[p,q] + c^2*A[q,q]
                A[p,q] <- A[q,p] <- 0

                // Update off-diagonal rows and columns r != p,q
                for r != p,q:
                    (A[r,p], A[r,q]) <- (c*A[r,p] - s*A[r,q],  s*A[r,p] + c*A[r,q])
                    A[p,r] <- A[r,p];  A[q,r] <- A[r,q]

                // Accumulate eigenvectors: V <- V * G
                for r = 0..n-1:
                    (V[r,p], V[r,q]) <- (c*V[r,p] - s*V[r,q],  s*V[r,p] + c*V[r,q])

    lambda <- diagonal of A
    return lambda, V  (sort ascending)
```

**Cost per sweep**: \f$\frac{n(n-1)}{2}\f$ rotations \f$\times\f$ \f$(4n + 4n)\f$ flops/rotation \f$= O(4n^3)\f$ per sweep.

### Convergence

The off-diagonal Frobenius norm satisfies:

\f[\sigma^2_{\text{off}}(A_{k+1}) \leq \left(1 - \frac{2}{n(n-1)}\right) \sigma^2_{\text{off}}(A_k) \qquad \text{(linear phase)}\f]

When \f$\sigma_{\text{off}}\f$ becomes small, convergence transitions to the **quadratic ultimate** phase:

\f[\sigma_{\text{off}}(A_{k+1}) \leq C\, \sigma^2_{\text{off}}(A_k)\f]

In practice: 5-10 sweeps for double precision with well-separated eigenvalues; <= 15 sweeps for clustered eigenvalues.

### Monotonicity of Diagonal Entries

Each rotation **decreases** \f$\sigma_{\text{off}}\f$ by exactly \f$2A_{pq}^2\f$:

\f[\sigma^2_{\text{off}}(A') = \sigma^2_{\text{off}}(A) - 2A_{pq}^2\f]

The Frobenius norm \f$\|A\|_F^2 = \sigma^2_{\text{diag}} + \sigma^2_{\text{off}}\f$ is preserved (similarity transform), so the diagonal Frobenius norm \f$\sigma^2_{\text{diag}} = \sum_i A_{ii}^2\f$ monotonically increases.

---

## Variants

### Threshold Jacobi

Skip rotation if \f$|A_{pq}| < \varepsilon_k \cdot \sigma_{\text{off}}\f$ (adaptive threshold). Reduces work in early sweeps when many entries are already small.

### One-Sided Jacobi

Apply rotations from the right only: \f$A_{k+1} = A_k G\f$. Column norms \f$\|A[:,j]\|\f$ converge to singular values. This is the basis of the one-sided Jacobi SVD -- see [svd-onesided-jacobi.md](svd-onesided-jacobi.md).

### Parallel Jacobi (Tournament Ordering)

The cyclic sweep visits pairs sequentially. Instead, use a **tournament schedule**: pair up all \f$n/2\f$ disjoint pairs simultaneously (no pair shares an index -> independent rotations). After \f$n-1\f$ rounds, all pairs have been visited:

\f[n = 8: \quad \text{Round 1}: (0,1),(2,3),(4,5),(6,7); \quad \text{Round 2}: (0,2),(1,4),(3,6),(5,7); \quad \ldots\f]

Each round requires a synchronization barrier, but rotations within a round are embarrassingly parallel -> GPU-friendly. This is the **Jacobi-Winograd** parallel scheme.

---

## Comparison with Other Eigensolvers

| Method | \f$A\f$ structure | Flop cost | Finds | Use when |
|--------|---------------|-----------|-------|----------|
| Cyclic Jacobi | Symmetric dense | \f$O(n^3 \times \text{sweeps})\f$ | All | Small \f$n\f$, high accuracy |
| Tridiagonalize + QR (LAPACK `dsyev`) | Symmetric dense | \f$O(\frac{4}{3}n^3)\f$ | All | Standard dense symmetric |
| Divide-and-conquer (`dstedc`) | Tridiagonal | \f$O(n^2)\f$ | All | After tridiagonalization |
| Power iteration | Any | \f$O(n^2) \times \f$ iters | 1 dominant | Sparse, dominant only |
| Lanczos | Symmetric sparse | \f$O(kn) \times \f$ iters | \f$k\f$ extreme | Large sparse, \f$k \ll n\f$ |

**LAPACK `dsyev`** uses:
1. Tridiagonalization via Householder: \f$\frac{4}{3}n^3\f$ flops (DSYTRD)
2. QR iteration on the tridiagonal: \f$O(n^2)\f$ flops (DSTEQR)
3. Back-transform eigenvectors: \f$2n^3\f$ flops (DORMTR)

Our Jacobi implementation is simpler and directly verifiable, but LAPACK's tridiagonalize-then-QR is 2-3\f$\times\f$ faster for large \f$n\f$.

---

## Performance Optimization

### Current Implementation

The rotation update has two phases:
- Diagonal update: 4 scalar flops
- Off-diagonal update: \f$4n\f$ flops (two length-\f$n\f$ rotation loops)
- Eigenvector accumulation: \f$4n\f$ flops

Total per rotation: \f$8n + 4 \approx 8n\f$ flops. Per sweep: \f$\frac{n(n-1)}{2} \times 8n = 4n^3(1 - 1/n)\f$ flops.

### SIMD for the Rotation Update (Column-Major A)

With **column-major** storage, columns \f$p\f$ and \f$q\f$ are contiguous. The off-diagonal update is a 2D rotation of two column vectors:

```cpp
__m256d vc = _mm256_broadcast_sd(&c);
__m256d vs_neg = _mm256_broadcast_sd(&s_neg);   // -s
__m256d vs = _mm256_broadcast_sd(&s);

for (idx r = 0; r < n; r += 4) {
    __m256d ap = _mm256_load_pd(&A(r, p));
    __m256d aq = _mm256_load_pd(&A(r, q));
    // new_p = c*ap - s*aq,  new_q = s*ap + c*aq
    __m256d new_p = _mm256_fmadd_pd(vc, ap, _mm256_mul_pd(vs_neg, aq));
    __m256d new_q = _mm256_fmadd_pd(vs, ap, _mm256_mul_pd(vc, aq));
    _mm256_store_pd(&A(r, p), new_p);
    _mm256_store_pd(&A(r, q), new_q);
}
```

Peak throughput: 2 FMA + 2 store = 10 instructions per 4 elements. For \f$n = 1024\f$: one rotation takes 256 FMA cycles ~= 85 ns at 3 GHz.

For **row-major** storage, exploit symmetry: update row \f$p\f$ and row \f$q\f$ of \f$A\f$ (same elements by symmetry). Keep a separate column-major buffer for \f$V\f$.

### Exploiting Symmetry in the Update

Store only the upper triangle of \f$A\f$ (since \f$A\f$ is symmetric). The rotation update for row \f$r\f$ (off-diagonal update) writes:

\f[A_{rp}' = c\, A_{rp} - s\, A_{rq}, \qquad A_{rq}' = s\, A_{rp} + c\, A_{rq}\f]

For \f$r < p\f$: access \f$A[r,p]\f$ and \f$A[r,q]\f$ (upper triangle) -- both accessible.
For \f$r > q\f$: access \f$A[p,r]\f$ and \f$A[q,r]\f$ (stored as \f$A[r,p]^T\f$) -- still accessible.

This halves memory traffic: update \f$n/2\f$ entries per rotation on average instead of \f$n\f$.

### Blocked Jacobi (Brent-Luk)

For large \f$n\f$, process rotations in **blocks** of size \f$b\f$. Instead of scalar Givens rotations in the \f$(p, q)\f$ plane, apply a \f$b \times b\f$ orthogonal transformation in the \f$(p:p+b, q:q+b)\f$ subspace:

1. Extract the \f$2b \times 2b\f$ subblock \f$B = \begin{pmatrix} A_{PP} & A_{PQ} \\ A_{QP} & A_{QQ} \end{pmatrix}\f$
2. Diagonalize \f$B\f$ via small \f$2b \times 2b\f$ Jacobi -> orthogonal matrix \f$G_{2b}\f$
3. Apply \f$G_{2b}\f$ to the full matrix: \f$A \leftarrow G^T A G\f$ as two DGEMM calls

The dominant cost becomes DGEMM (BLAS-3): arithmetic intensity \f$O(b)\f$ instead of \f$O(1)\f$ for scalar FMA. With \f$b = 64\f$ and \f$n = 1024\f$: achieve near-peak GFLOP/s on the trailing DGEMM portion.

### Cache Behaviour

The rotation simultaneously accesses columns \f$p\f$ and \f$q\f$. With column-major storage, both are contiguous: \f$2n \times 8\f$ bytes. For \f$n = 2048\f$: each column is 16 KB, two columns = 32 KB -> fits in L1 (32 KB typical). For \f$n = 4096\f$: 64 KB -> L2 traffic.

**Column-pair buffer**: copy columns \f$p, q\f$ to 32-byte-aligned scratch buffers, update scratch, write back. Eliminates potential L1 conflict misses when \f$p, q\f$ map to the same cache set.
