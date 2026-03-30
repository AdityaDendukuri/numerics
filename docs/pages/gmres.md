# GMRES -- Generalized Minimal Residual Method {#page_gmres_notes}

## Problem Statement

Solve \f$Ax = b\f$ where \f$A \in \mathbb{R}^{n \times n}\f$ is non-singular but possibly **non-symmetric**. CG requires symmetry and positive definiteness; GMRES works for any invertible \f$A\f$.

GMRES finds the iterate \f$x_k \in x_0 + \mathcal{K}_k(A, r_0)\f$ that **minimizes the 2-norm of the residual** over the Krylov subspace:

\f[x_k = \operatorname*{argmin}_{x \in x_0 + \mathcal{K}_k} \|Ax - b\|_2, \qquad \mathcal{K}_k = \operatorname{span}\!\bigl\{r_0, Ar_0, \ldots, A^{k-1}r_0\bigr\}\f]

The basis grows by one vector per iteration. **Restarted GMRES(\f$m\f$)** caps memory at \f$m\f$ vectors by restarting from the current iterate every \f$m\f$ steps.

---

## Arnoldi Process

The Krylov basis is built via the **Arnoldi recurrence** using Modified Gram-Schmidt:

```
function Arnoldi(A, v_j, V, H, j):
    w <- A v_j

    // Modified Gram-Schmidt against previous basis vectors
    for i = 0 to j:
        H[j,i] <- w^T V[i]
        w      <- w - H[j,i] * V[i]

    H[j, j+1] <- ||w||_2
    v_{j+1}   <- w / H[j, j+1]    // next basis vector
```

After \f$k\f$ steps this produces the **Arnoldi relation**:

\f[A V_k = V_{k+1} \bar{H}_k\f]

where \f$V_k = [v_1 \mid \cdots \mid v_k]\f$ has orthonormal columns and \f$\bar{H}_k \in \mathbb{R}^{(k+1) \times k}\f$ is the upper Hessenberg matrix of Gram-Schmidt coefficients.

---

## The GMRES Least-Squares Problem

The residual at step \f$k\f$ satisfies:

\f[\|Ax_k - b\|_2 = \|A(x_0 + V_k y_k) - b\|_2 = \|AV_k y_k - r_0\|_2\f]

Using the Arnoldi relation and the fact that \f$V_{k+1}\f$ is orthonormal:

\f[\|Ax_k - b\|_2 = \|V_{k+1}\bar{H}_k y_k - \beta e_1\|_2 = \|\bar{H}_k y_k - \beta e_1\|_2, \quad \beta = \|r_0\|_2\f]

This is a small \f$(k+1) \times k\f$ least-squares problem -- solved via Givens rotations applied **incrementally** (one new rotation per step):

\f[\begin{pmatrix} c_j & s_j \\ -s_j & c_j \end{pmatrix} \begin{pmatrix} H_{jj} \\ H_{j,j+1} \end{pmatrix} = \begin{pmatrix} r \\ 0 \end{pmatrix}, \qquad c_j = \frac{H_{jj}}{d},\ s_j = \frac{H_{j,j+1}}{d},\ d = \sqrt{H_{jj}^2 + H_{j,j+1}^2}\f]

The rotated RHS vector \f$g\f$ gives: \f$|g_{j+1}| = \|r_{j+1}\|_2\f$ -- monitor this for convergence **without** computing \f$x_k\f$ explicitly.

---

## Algorithm: Restarted GMRES(\f$m\f$)

```
function GMRES(A, b, x, tol, max_iter, restart):
    while total_iters < max_iter:
        r <- b - Ax;  beta <- ||r||_2
        if beta < tol: return (x, converged)

        V[0] <- r / beta
        g[0] <- beta;  g[1:m+1] <- 0

        for j = 0 to restart-1:
            // Arnoldi step (MGS)
            w <- A V[j]
            for i = 0 to j:
                H[j,i] <- w^T V[i];  w <- w - H[j,i] * V[i]
            H[j,j+1] <- ||w||;  if H[j,j+1] < eps: break
            V[j+1] <- w / H[j,j+1]

            // Apply previous Givens rotations to column j of H
            for i = 0 to j-1:
                [H[j,i], H[j,i+1]] <- G_i [H[j,i], H[j,i+1]]

            // New Givens rotation to zero H[j,j+1]
            d <- sqrt(H[j,j]^2 + H[j,j+1]^2)
            c[j] <- H[j,j]/d;  s[j] <- H[j,j+1]/d
            H[j,j] <-  c[j]*H[j,j] + s[j]*H[j,j+1];  H[j,j+1] <- 0
            g[j+1] <- -s[j]*g[j];  g[j] <- c[j]*g[j]

            if |g[j+1]| < tol: goto back_solve

        back_solve:
            m_actual <- j+1
            // Back-solve upper triangular H[0:m,0:m] y = g[0:m]
            y[m-1] <- g[m-1] / H[m-1,m-1]
            for i = m-2 downto 0:
                y[i] <- (g[i] - sum_{k=i+1}^{m-1} H[k,i]*y[k]) / H[i,i]

            x <- x + sum_{i=0}^{m-1} y[i] * V[i]

        if converged: break
```

**Memory**: \f$(m+1)\f$ vectors of length \f$n\f$ -> \f$O(mn)\f$ storage. Typical restart: \f$m \in [30, 300]\f$.

**Flops per restart**: \f$2mn^2\f$ (matvecs) \f$+ 2m^2n\f$ (MGS) \f$+ O(m^2)\f$ (Givens). For \f$m \ll n\f$, the matvec dominates.

---

## Convergence

For \f$A = I + E\f$ with \f$\|E\|\f$ small, GMRES converges like CG in \f$O(\sqrt{\kappa})\f$ iterations. In general, GMRES convergence is governed by the polynomial approximation problem:

\f[\frac{\|r_k\|_2}{\|r_0\|_2} \leq \min_{\substack{p_k \in \mathcal{P}_k \\ p_k(0) = 1}} \max_{\lambda \in \sigma(A)} |p_k(\lambda)|\f]

- **Normal \f$A\f$** (i.e., \f$A = U\Lambda U^H\f$): reduces to approximation of zero on the spectrum, similar to CG
- **Clustered spectrum**: fast convergence -- a low-degree polynomial can be small on a few eigenvalue clusters
- **Stagnation**: GMRES can plateau many steps then converge suddenly (e.g., one outlier eigenvalue far from the main cluster)

### Preconditioning

**Left preconditioning**: solve \f$M^{-1}Ax = M^{-1}b\f$:

\f[\text{GMRES on } M^{-1}A \text{ minimizes } \|M^{-1}r_k\|_2\f]

**Right preconditioning**: solve \f$AM^{-1}(Mx) = b\f$, then \f$x \leftarrow M^{-1}(Mx)\f$:

\f[\text{GMRES on } AM^{-1} \text{ minimizes } \|r_k\|_2 \quad \text{(true residual)}\f]

Right preconditioning is usually preferred because it minimizes the physically meaningful residual norm. Left preconditioning minimizes \f$\|M^{-1}r\|_2\f$ which depends on \f$M\f$.

Common preconditioners for non-symmetric problems:
- **ILU(\f$k\f$)**: incomplete LU with level-\f$k\f$ fill -- general, moderate improvement
- **Block Jacobi**: partition rows into blocks, invert each block exactly -- parallelizes
- **SSOR**: symmetric successive over-relaxation -- cheap, effective for M-matrices
- **AMG**: algebraic multigrid -- near \f$O(n)\f$ convergence for elliptic PDE

---

## Variants

| Variant | Memory | Notes |
|---------|--------|-------|
| Full GMRES | \f$O(n^2)\f$ | No restart; \f$n\f$-step convergence; impractical for large \f$n\f$ |
| GMRES(\f$m\f$) | \f$O(mn)\f$ | Restart every \f$m\f$ steps; our implementation |
| FGMRES | \f$2O(mn)\f$ | Flexible preconditioner (can change each step) |
| LGMRES | \f$O(mn)\f$ | Augment Krylov space with error vectors from previous restarts |
| BiCGSTAB | \f$O(n)\f$ | Short recurrence; non-symmetric; not min-residual but cheaper/restart-free |
| MINRES | \f$O(n)\f$ | Symmetric indefinite \f$A\f$; short recurrence; exact residual minimization |

---

## Performance Optimization

### BLAS-2 Arnoldi

Standard MGS is a sequence of BLAS-1 DOT + AXPY at step \f$j\f$: \f$O(j \cdot n)\f$ cost, \f$O(m^2 n)\f$ total. Replace with BLAS-2 DGEMV:

```
// All j+1 dot products at once:
h = dgemv(V[:,0:j+1]^T, w)          // (j+1) x n  times  n x 1 -> O(jn), one BLAS call
w -= dgemv(V[:,0:j+1], h)            // n x (j+1)  times  (j+1) x 1
```

Replaces \f$j\f$ DDOT \f$+ j\f$ DAXPY calls with 2 DGEMV calls, improving cache efficiency (\f$V[:,0:j+1]\f$ loaded once for both operations).

### Block Arnoldi (BLAS-3)

Orthogonalize \f$b\f$ vectors simultaneously:

\f[[v_{j+1}, \ldots, v_{j+b}] = AV[:,j:j+b] \quad \to \quad \text{DGEMM}\f]
\f[H \leftarrow V[:,0:j]^T W, \quad W \leftarrow W - V[:,0:j] H \quad \to \quad \text{2 DGEMM calls}\f]

This converts BLAS-1/2 operations to BLAS-3 DGEMM, dramatically improving arithmetic intensity on cache-blocked hardware.

### Communication-Avoiding GMRES (CA-GMRES)

Standard GMRES performs one global all-reduce per iteration for \f$\|w\|\f$ in MGS. For \f$p\f$ distributed processes, this is \f$O(\log p)\f$ latency per step.

CA-GMRES computes \f$s\f$ basis vectors per communication round using matrix powers:

\f[[v,\, Av,\, A^2v,\, \ldots,\, A^s v]\f]

(a Newton or Chebyshev basis), then orthogonalizes the block. This reduces all-reduces by factor \f$s\f$ at the cost of potential loss of numerical orthogonality (mitigated by careful basis selection).

### SIMD for Dense Matvec and MGS

**Matvec** (row \f$i\f$ dot with \f$p\f$):
```cpp
__m256d acc = _mm256_setzero_pd();
for (idx j = 0; j < n; j += 4)
    acc = _mm256_fmadd_pd(_mm256_load_pd(&A(i,j)),
                           _mm256_load_pd(&p[j]), acc);
w[i] = hsum(acc);
```

**Fused dot + AXPY for MGS** (two passes required since \f$h\f$ is needed for the update):

```cpp
// Pass 1: h = w^T V[i]
__m256d vdot = _mm256_setzero_pd();
for (idx k = 0; k < n; k += 4)
    vdot = _mm256_fmadd_pd(_mm256_load_pd(&w[k]),
                            _mm256_load_pd(&V[i][k]), vdot);
real h = hsum(vdot);

// Pass 2: w -= h * V[i]
__m256d vh = _mm256_broadcast_sd(&h);
for (idx k = 0; k < n; k += 4) {
    __m256d vw = _mm256_load_pd(&w[k]);
    __m256d vv = _mm256_load_pd(&V[i][k]);
    _mm256_store_pd(&w[k], _mm256_fnmadd_pd(vh, vv, vw));
}
```

**Delayed reorthogonalization**: compute all \f$h_{ij}\f$ values in one matvec scan (one pass over each \f$V[:,i]\f$), then apply all updates in a second scan. Improves cache use when \f$m\f$ is large.
