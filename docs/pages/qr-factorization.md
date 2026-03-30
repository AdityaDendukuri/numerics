# QR Factorization via Householder Reflections {#page_qr_notes}

## Mathematical Background

Every matrix \f$A \in \mathbb{R}^{m \times n}\f$ with \f$m \geq n\f$ admits the decomposition

\f[A = QR\f]

where \f$Q \in \mathbb{R}^{m \times m}\f$ is orthogonal (\f$Q^TQ = I\f$) and \f$R \in \mathbb{R}^{m \times n}\f$ is upper triangular. The *economy* (thin) QR uses \f$\hat{Q} \in \mathbb{R}^{m \times n}\f$ and \f$\hat{R} \in \mathbb{R}^{n \times n}\f$.

QR is fundamental for:
- **Least-squares**: \f$\min_x \|Ax - b\|_2\f$ with \f$m > n\f$ (overdetermined)
- **Eigenvalue algorithms**: QR iteration for non-symmetric \f$A\f$
- **Krylov orthogonalization**: Gram-Schmidt in GMRES and Lanczos

---

## Householder Reflections

A Householder reflector is the rank-1 symmetric orthogonal matrix

\f[H = I - \frac{2vv^T}{v^Tv}, \qquad H^T = H, \quad H^2 = I, \quad \det(H) = -1\f]

For a vector \f$x \in \mathbb{R}^m\f$, choose \f$v\f$ so that \f$H\f$ maps \f$x\f$ to a scalar multiple of \f$e_1\f$:

\f[Hx = -\operatorname{sign}(x_1)\|x\|_2\, e_1\f]

**Construction**:

\f[\sigma \leftarrow \operatorname{sign}(x_1)\|x\|_2, \qquad v \leftarrow x + \sigma e_1, \qquad v \leftarrow v/\|v\|\f]

**Sign convention**: choose the same sign as \f$x_1\f$. This avoids catastrophic cancellation: if \f$x_1 > 0\f$, then \f$v_1 = x_1 + \|x\| > 0\f$ is a sum of positives. The alternative \f$v_1 = x_1 - \|x\|\f$ would subtract nearly equal numbers when \f$x_1 \approx \|x\|\f$.

---

## Algorithm

Apply \f$r = \min(m-1, n)\f$ reflections. Step \f$k\f$ zeros all entries of column \f$k\f$ below the diagonal.

```
function QR(A):
    m, n <- A.rows, A.cols
    r <- min(m-1, n)
    R <- copy of A
    vs <- list of r Householder vectors

    for k = 0 to r-1:
        x <- R[k:m, k]                       // column stub (length m-k)
        sigma <- sign(x[0]) * ||x||
        v <- x;  v[0] += sigma               // unnormalised Householder vector
        v /= ||v||;  vs[k] <- v

        // Apply H_k = I - 2vv^T to trailing submatrix R[k:m, k:n]
        // For each column j: R[k:m,j] -= 2v * (v^T R[k:m,j])
        for j = k to n-1:
            tau <- 2 * (v^T R[k:m, j])
            R[k:m, j] -= tau * v

    // Build Q by accumulating reflectors in reverse order
    Q <- I_m
    for k = r-1 downto 0:
        v <- vs[k]
        for j = k to m-1:        // columns j < k are already correct (= e_j)
            tau <- 2 * (v^T Q[k:m, j])
            Q[k:m, j] -= tau * v

    // Zero sub-diagonal of R (kill floating-point noise)
    R[i,j] <- 0  for all i > j

    return (Q, R)
```

**Complexity**:
- Factorization: \f$2mn^2 - \frac{2}{3}n^3\f$ flops
- Building \f$Q\f$: \f$4m^2n - 2mn^2 + \frac{2}{3}n^3\f$ flops (dominant for \f$m \gg n\f$)

If only \f$Q^Tb\f$ is needed (least-squares), **never build \f$Q\f$ explicitly** -- apply each \f$H_k\f$ to \f$b\f$ in \f$O(mn)\f$ total.

---

## Least-Squares Solve

\f$\min_x \|Ax - b\|_2\f$ has normal equations \f$A^TAx = A^Tb\f$, but solving those squares the condition number. The QR approach is numerically preferable:

\f[\|Ax - b\|_2^2 = \|QRx - b\|_2^2 = \|Rx - Q^Tb\|_2^2 = \|\hat{R}x - \tilde{c}\|_2^2 + \|\bar{c}\|_2^2\f]

where \f$c = Q^Tb\f$, \f$\tilde{c} = c[0:n]\f$ (matched components), \f$\bar{c} = c[n:m]\f$ (residual).

```
function QR_solve(Q, R, b):
    c <- Q^T b           // O(mn): matrix-vector with Q transposed
    x <- back-solve R[0:n, 0:n] x = c[0:n]   // O(n^2)
    return x
```

---

## Stability

QR via Householder is **backward stable**: the computed solution \f$\hat{x}\f$ satisfies

\f[(A + \delta A)\hat{x} = b, \qquad \frac{\|\delta A\|_F}{\|A\|_F} \leq O(\varepsilon_{\text{mach}})\f]

independent of \f$\kappa(A)\f$. The normal equations satisfy \f$\|\delta A\| / \|A\| \leq O(\varepsilon_{\text{mach}}\,\kappa(A)^2)\f$ -- for \f$\kappa(A) > 10^8\f$ in double precision, the normal equations lose all significant digits while QR still gives correct residuals.

---

## Alternatives: Givens vs MGS vs Householder

| Method | Flops | Stability | Use case |
|--------|-------|-----------|----------|
| Householder | \f$2mn^2 - \frac{2}{3}n^3\f$ | Backward stable | Dense QR, standard |
| Modified Gram-Schmidt (MGS) | \f$2mn^2\f$ | Conditionally stable | Krylov (Arnoldi), thin \f$Q\f$ |
| Classical Gram-Schmidt (CGS) | \f$2mn^2\f$ | Unstable | Never use alone |
| Givens rotations | \f$\sim 3mn^2\f$ | Backward stable | Banded/sparse, rank-1 updates |

**Givens** is preferred when \f$A\f$ has structured sparsity: each rotation \f$G(i,j,\theta)\f$ modifies only rows \f$i\f$ and \f$j\f$, zeroing one element without disturbing the rest. For dense \f$A\f$, Householder is preferred -- one reflector zeros an entire column stub.

---

## Performance Optimization

### Current Implementation

The apply-reflector inner loop processes columns one at a time:

```
for j = k to n-1:
    tau <- 2 * (v^T R[k:m, j])     // dot product: length m-k
    R[k:m, j] -= tau * v           // AXPY: length m-k
```

With row-major storage, `R[k:m, j]` (a column) is strided -- every access is a cache miss for large \f$m\f$. This is the dominant cost for tall matrices.

### Blocked Householder: WY Representation

Accumulate a block of \f$n_b\f$ Householder vectors into a compact WY form:

\f[H_1 H_2 \cdots H_{n_b} = I - WY^T, \qquad W, Y \in \mathbb{R}^{m \times n_b}\f]

Apply the entire block to the trailing submatrix in one `DGEMM`:

\f[A \leftarrow A - W(Y^T A) \qquad \text{(2 DGEMM calls)}\f]

This is a BLAS-3 operation with arithmetic intensity \f$O(n_b)\f$. LAPACK uses this in `dgeqrf` with \f$n_b \approx 64\f$.

**Building the WY representation**: \f$O(m \times n_b^2)\f$ -- the same as the unblocked panel factorization, negligible relative to the trailing DGEMM.

### SIMD for the AXPY Kernel

The per-column AXPY `R[k:m,j] -= tau * v` for length \f$\ell = m - k\f$:

```cpp
__m256d vtau = _mm256_broadcast_sd(&tau);
for (idx i = 0; i < len; i += 4) {
    __m256d vv = _mm256_load_pd(v + i);
    __m256d vr = _mm256_load_pd(R_col + i);
    vr = _mm256_fnmadd_pd(vtau, vv, vr);      // r -= tau * v
    _mm256_store_pd(R_col + i, vr);
}
```

For strided column access (row-major \f$A\f$): **transpose the panel** into a column-major buffer before applying reflectors, then transpose back. This converts \f$O(mn/4)\f$ gather operations into \f$O(m)\f$ sequential reads.

### Avoiding Explicit Q

In practice \f$Q\f$ is almost never needed. Store the compact reflectors and apply \f$Q^T\f$ on-demand in \f$O(mn)\f$:

```
// Apply Q^T to vector b without forming Q
for k = 0 to r-1:
    tau <- 2 * (vs[k]^T b[k:m])
    b[k:m] -= tau * vs[k]
```

This halves memory usage (no \f$m \times m\f$ matrix) and streams through \f$A\f$ once.

### Tall-Skinny QR (TSQR) for Distributed Systems

For \f$m \gg n\f$ (e.g., \f$m = 10^6\f$, \f$n = 100\f$), use a communication-avoiding tree reduction:

1. Partition \f$A\f$ into \f$p\f$ row blocks: \f$A = [A_1;\, A_2;\, \ldots;\, A_p]\f$
2. Factor each \f$A_i \to Q_i R_i\f$ in parallel (each block fits in L2)
3. Stack the \f$R\f$ blocks: \f$[R_1;\, R_2;\, \ldots;\, R_p] \xrightarrow{\text{QR}} R\f$ (small \f$p \times n\f$ problem)

TSQR achieves the same backward stability as standard Householder with communication volume \f$O(n^2 \log p)\f$ vs \f$O(mn)\f$ for column-panel QR. It is the basis for Communication-Avoiding QR (CAQR) in ScaLAPACK and SLATE.
