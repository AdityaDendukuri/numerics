# Conjugate Gradient and the Thomas Algorithm {#page_cg_notes}

## Part I -- Conjugate Gradient (CG)

### Problem Statement

Solve \f$Ax = b\f$ where \f$A \in \mathbb{R}^{n \times n}\f$ is **symmetric positive definite** (SPD). CG minimizes the \f$A\f$-norm of the error over the \f$k\f$-dimensional Krylov subspace:

\f[\mathcal{K}_k(A, r_0) = \operatorname{span}\!\bigl\{r_0,\, Ar_0,\, A^2r_0,\, \ldots,\, A^{k-1}r_0\bigr\}\f]

The iterate \f$x_k\f$ is the unique point in \f$x_0 + \mathcal{K}_k\f$ minimizing \f$\|x - x^*\|_A = \sqrt{(x-x^*)^T A (x-x^*)}\f$.

### Derivation

Minimizing \f$\|e\|_A\f$ is equivalent to minimizing the quadratic form

\f[\varphi(x) = \tfrac{1}{2}x^TAx - b^Tx, \qquad \nabla\varphi = Ax - b = -r\f]

At each step the new search direction \f$p_k\f$ is made **\f$A\f$-conjugate** to all previous directions: \f$p_k^T A p_j = 0\f$ for \f$j < k\f$. This allows minimization of \f$\varphi\f$ along \f$p_k\f$ without disturbing progress already made along \f$p_0, \ldots, p_{k-1}\f$.

### Algorithm

```
function CG(A, b, x_0, tol, max_iter):
    r <- b - A x_0
    p <- r
    rho_old <- r^T r

    for iter = 1, 2, ..., max_iter:
        w   <- A p                          // matvec: O(n^2) dense, O(nnz) sparse
        alpha <- rho_old / (p^T w)          // step length
        x   <- x + alpha * p
        r   <- r - alpha * w
        rho_new <- r^T r

        if sqrt(rho_new) < tol: return x, converged

        beta <- rho_new / rho_old           // Gram-Schmidt coefficient
        p    <- r + beta * p                // new A-conjugate direction
        rho_old <- rho_new
```

**Flops per iteration**: \f$2n^2\f$ (matvec) \f$+ 6n\f$ (dot, axpy, scale) \f$\approx 2n^2\f$ for large \f$n\f$.

**Memory**: vectors \f$r, p, w\f$ plus the matrix -- \f$O(n^2)\f$ for dense \f$A\f$, or \f$O(\text{nnz})\f$ for sparse. CG never modifies \f$A\f$.

### Convergence

CG terminates in at most \f$n\f$ iterations in exact arithmetic. In floating-point, convergence requires \f$O(\sqrt{\kappa})\f$ iterations where \f$\kappa = \kappa_2(A) = \lambda_{\max}/\lambda_{\min}\f$:

\f[\frac{\|e_k\|_A}{\|e_0\|_A} \leq 2\left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^k\f]

For poorly conditioned \f$A\f$ (\f$\kappa \gg 1\f$), CG stagnates without preconditioning.

**Eigenvalue clustering** accelerates convergence: if all eigenvalues except one small cluster lie in a single interval, CG quickly annihilates the polynomial responsible for those eigenvalues and then reduces to a problem of size equal to the cluster count.

### Preconditioned CG (PCG)

Replace \f$A\f$ with \f$M^{-1}A\f$ where \f$M \approx A\f$ is cheap to invert. The modified iteration solves \f$M^{-1/2} A M^{-1/2} \tilde{y} = M^{-1/2}b\f$, which has condition number \f$\kappa(M^{-1}A)\f$.

```
function PCG(A, M_inv, b, x, tol):
    r <- b - Ax
    z <- M_inv(r)          // preconditioner solve
    p <- z
    rho_old <- r^T z       // note: r^T z, not r^T r

    for iter = ...:
        w     <- Ap
        alpha <- rho_old / (p^T w)
        x <- x + alpha*p;  r <- r - alpha*w
        z     <- M_inv(r)
        rho_new <- r^T z
        beta  <- rho_new / rho_old
        p     <- z + beta*p
        rho_old <- rho_new
```

Common preconditioners:
- **Diagonal (Jacobi)**: \f$M = \operatorname{diag}(A)\f$ -- always cheap, effective when \f$A\f$ is diagonally dominant
- **Incomplete Cholesky (IC0)**: \f$M = \hat{L}\hat{L}^T\f$ where \f$\hat{L}\f$ has the same sparsity as \f$\operatorname{tril}(A)\f$
- **Algebraic multigrid (AMG)**: \f$\kappa(M^{-1}A) \approx O(1)\f$ for Poisson-type problems

### Matrix-Free CG

`cg_matfree` accepts a `MatVecFn` callback instead of an explicit matrix, enabling:
- Implicit time stepping (\f$A = M + \Delta t\, K\f$, never assembled)
- Spectral methods (\f$A\f$ applied via FFT in \f$O(n \log n)\f$)
- Structured operators (Toeplitz, circulant -- \f$O(n)\f$ storage)

---

## Part II -- Thomas Algorithm (Tridiagonal Solver)

### Problem Statement

Solve the tridiagonal system \f$Tx = d\f$ where

\f[T = \begin{pmatrix} b_0 & c_0 & & \\ a_1 & b_1 & c_1 & \\ & a_2 & b_2 & \ddots \\ & & \ddots & \ddots & c_{n-2} \\ & & & a_{n-1} & b_{n-1} \end{pmatrix}\f]

This is LU factorization specialized to tridiagonal structure: \f$O(n)\f$ work instead of \f$O(n^2)\f$ for dense forward elimination.

### Algorithm

```
function Thomas(a, b, c, d):
    // a: sub-diagonal (length n-1)
    // b: main diagonal (length n, modified in-place)
    // c: super-diagonal (length n-1)
    // d: RHS (length n, modified in-place)

    // Forward sweep: eliminate sub-diagonal
    for i = 1 to n-1:
        w    <- a[i] / b[i-1]
        b[i] <- b[i] - w * c[i-1]
        d[i] <- d[i] - w * d[i-1]

    // Backward substitution
    x[n-1] <- d[n-1] / b[n-1]
    for i = n-2 downto 0:
        x[i] <- (d[i] - c[i] * x[i+1]) / b[i]

    return x
```

**Complexity**: \f$8n\f$ flops.

**Stability**: unconditionally stable when \f$A\f$ is SPD or strictly diagonally dominant (\f$|b_i| > |a_i| + |c_i|\f$ for all \f$i\f$).

### Applicability

Tridiagonal systems arise everywhere in numerical PDE:
- 1D finite difference / finite element discretization
- Implicit time integration (Crank-Nicolson for the heat equation \f$u_t = u_{xx}\f$)
- Cubic spline interpolation (natural and clamped boundary conditions)
- ADI (alternating direction implicit) splitting for 2D/3D parabolic equations

### Block Tridiagonal Systems

In 2D finite differences, the system is block tridiagonal with \f$n\f$ blocks of size \f$n\f$:

\f[\begin{pmatrix} B & C & & \\ A & B & C & \\ & \ddots & \ddots & \ddots \\ & & A & B \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{pmatrix} = \begin{pmatrix} d_1 \\ d_2 \\ \vdots \\ d_n \end{pmatrix}\f]

The same Thomas algorithm applies with block BLAS operations: `DGEMM` replaces scalar multiply, `DTRSM` replaces divide. Full block Thomas costs \f$O(n^3) \times n = O(n^4)\f$ for an \f$n \times n\f$ 2D grid.

---

## Performance Optimization

### CG: SIMD for Dense Matvec

The dominant cost in each CG iteration is the matvec \f$w \leftarrow Ap\f$. For row-major dense \f$A\f$:

```cpp
for (idx i = 0; i < n; ++i) {
    __m256d acc = _mm256_setzero_pd();
    for (idx j = 0; j < n; j += 4)
        acc = _mm256_fmadd_pd(_mm256_load_pd(&A(i,j)),
                               _mm256_load_pd(&p[j]), acc);
    w[i] = hsum(acc);
}
```

FMA throughput on AVX2: \f$2 \times 4 = 8\f$ flops/cycle. For large \f$n\f$, performance is memory-bandwidth limited.

### CG: Pipelining (Chronopoulos-Gear)

Standard CG requires two global synchronizations per iteration (one for \f$\rho = r^Tr\f$ and one for \f$\sigma = p^TAp\f$). The **Chronopoulos-Gear** variant (s-step CG) computes \f$r^Tr\f$ and \f$p^TAp\f$ in the same matvec loop:

\f[\sigma \leftarrow \sum_i w_i p_i, \quad \rho \leftarrow \sum_i r_i^2 \qquad \text{computed simultaneously in one pass over } w, p, r\f]

This halves memory bandwidth and reduces synchronization -- critical for distributed-memory CG.

### Thomas: SIMD over Multiple RHS

Thomas is memory-bandwidth bound: each element is loaded and stored once. For \f$m\f$ independent RHS, unroll over RHS with SIMD -- process 4 or 8 simultaneously:

```cpp
// Forward sweep: process 4 RHS at once
for (idx i = 1; i < n; ++i) {
    real w = a[i] / b_work[i-1];
    b_work[i] -= w * c[i-1];             // scalar (shared b)
    // Vectorised over 4 RHS:
    __m256d vw = _mm256_broadcast_sd(&w);
    __m256d di   = _mm256_load_pd(&D(i,   0));
    __m256d dim1 = _mm256_load_pd(&D(i-1, 0));
    _mm256_store_pd(&D(i, 0), _mm256_fnmadd_pd(vw, dim1, di));
}
```

Peak Thomas throughput (bandwidth-limited): \f$\sim 100\,\text{GFLOP/s}\f$ on modern hardware with 100 GB/s DRAM.

### Cyclic Reduction (CR) -- Parallel Alternative

CR achieves \f$O(n \log n)\f$ work with \f$O(\log n)\f$ parallel depth:

```
for level = 0 to log2(n) - 1:
    step <- 2^(level+1)
    for i = step/2, 3*step/2, ...:  (in parallel)
        eliminate rows i - step/2 and i + step/2 from row i
// After log2(n) levels: one equation per processor
// Back-substitute in reverse
```

CR is ideal for GPU (CUDA): all eliminations at each level are independent -> \f$O(n/\text{step})\f$ parallel threads. NVIDIA cuSPARSE uses a hybrid CR-PCR (parallel cyclic reduction) approach.
