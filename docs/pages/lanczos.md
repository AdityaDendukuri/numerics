# Lanczos Algorithm with Full Reorthogonalization {#page_lanczos_notes}

## Problem Statement

Given a **symmetric** matrix \f$A \in \mathbb{R}^{n \times n}\f$ (or a symmetric linear operator defined by a matvec), find the \f$k\f$ dominant or extreme eigenpairs efficiently when \f$k \ll n\f$. Lanczos builds a \f$k\f$-dimensional Krylov subspace containing near-optimal approximations to the extreme eigenvalues.

Applications: graph Laplacian spectral analysis, quantum chemistry (CI), structural vibration modes, PCA via SVD, spectral graph clustering.

---

## The Lanczos Recurrence

Starting from a unit vector \f$v_1\f$, the \f$k\f$-step Lanczos process builds an orthonormal basis \f$V_k = [v_1, \ldots, v_k]\f$ for the Krylov subspace \f$\mathcal{K}_k(A, v_1)\f$ satisfying:

\f[A V_k = V_k T_k + \beta_k v_{k+1} e_k^T\f]

where \f$T_k\f$ is the symmetric **tridiagonal** matrix:

\f[T_k = \begin{pmatrix} \alpha_1 & \beta_1 & & \\ \beta_1 & \alpha_2 & \beta_2 & \\ & \ddots & \ddots & \ddots \\ & & \beta_{k-1} & \alpha_k \end{pmatrix}\f]

The three-term recurrence generating each new basis vector:

\f[w = A v_j, \qquad \alpha_j = v_j^T w, \qquad w \leftarrow w - \alpha_j v_j - \beta_{j-1} v_{j-1}, \qquad \beta_j = \|w\|, \qquad v_{j+1} = w/\beta_j\f]

This is the **Lanczos relation**. In exact arithmetic, the three-term recurrence alone maintains orthogonality. In floating-point, rounding errors cause loss of orthogonality -> ghost eigenvalues -> **full reorthogonalization is required**.

---

## Algorithm: Lanczos with Full Reorthogonalization (MGS)

```
function lanczos(matvec, n, k, tol, max_steps):
    max_steps <- min(3k, n)   if not specified
    V[n x max_steps] <- 0
    alpha[max_steps] <- 0
    beta[max_steps]  <- 0

    V[:,0] <- e_1              // deterministic start

    for j = 0 to max_steps-1:
        vj <- V[:,j]
        w  <- matvec(vj)       // dominant cost: O(n^2) dense, O(nnz) sparse

        // Three-term recurrence
        alpha[j] <- vj^T w
        w <- w - alpha[j]*vj
        if j > 0: w <- w - beta[j-1] * V[:,j-1]

        // Full reorthogonalization: project w onto orthogonal complement of span{v_0,...,v_j}
        // \f$w \leftarrow w - \sum_{l=0}^{j} (v_l^T w)\, v_l\f$
        for l = 0 to j:
            proj <- V[:,l]^T w
            w    <- w - proj * V[:,l]

        beta[j] <- ||w||
        if beta[j] < 1e-12: break          // invariant subspace

        if j+1 < max_steps: V[:,j+1] <- w / beta[j]

    // Build m x m tridiagonal T (m = steps taken)
    T[j,j] <- alpha[j];  T[j,j+1] = T[j+1,j] <- beta[j]

    // Eigendecompose T: O(m^3) -- negligible for m << n
    (lambda_tilde, S) <- eig_sym(T)

    // Ritz vectors: V_m * (columns of S), cost O(mn)
    U_tilde[r,i] = sum_j S[j,i] * V[r,j]

    // Convergence check: ||A u_i - lambda_i u_i||_2 for each Ritz pair
    return (lambda_tilde, U_tilde, steps, converged)
```

**Total cost**: \f$O(k \cdot \text{nnz})\f$ (matvecs) \f$+ O(k^2 n)\f$ (reorthogonalization) \f$+ O(k^3)\f$ (inner eigensolver, negligible).

---

## The Projection Principle

The tridiagonal \f$T_k\f$ is the **Galerkin projection** of \f$A\f$ onto \f$\mathcal{K}_k\f$:

\f[T_k = V_k^T A V_k \in \mathbb{R}^{k \times k}\f]

Its eigenvalues -- the **Ritz values** \f$\theta_1, \ldots, \theta_k\f$ -- are the best polynomial approximations to \f$A\f$'s eigenvalues within \f$\mathcal{K}_k\f$. Specifically, \f$\theta_i\f$ minimizes \f$|p(\lambda_i)|^2\f$ over all degree-\f$k\f$ polynomials \f$p\f$ with \f$p(0) = 1\f$ in the subspace sense.

The Ritz vectors \f$\tilde{u}_i = V_k s_i\f$ (columns of \f$V_k S\f$) satisfy:

\f[\|A\tilde{u}_i - \theta_i \tilde{u}_i\|_2 = \beta_k |S_{k,i}| \qquad \text{(cheap convergence estimate!)}\f]

**Convergence order**: extreme eigenvalues (\f$\lambda_1\f$, \f$\lambda_n\f$) converge first; interior eigenvalues last.

---

## Convergence

Convergence is governed by the **spectral gap** \f$\delta = (\lambda_1 - \lambda_2)/(\lambda_1 - \lambda_n)\f$:
- Large gap (\f$\delta \gg 0\f$): leading Ritz value converges in \f$O(1/\sqrt{\delta})\f$ steps
- Clustered extreme eigenvalues: slow convergence, need large \f$k\f$

The \f$\beta_k |S_{k,i}|\f$ residual estimate from the algorithm provides an **inexpensive stopping criterion** -- no need to compute the full Ritz vector to check convergence.

---

## Ghost Eigenvalues and Reorthogonalization

In floating-point, the three-term recurrence loses orthogonality after many steps. Once a Ritz value converges to \f$\lambda_j\f$, the Lanczos vector drifts back into the \f$u_j\f$ eigenspace, causing \f$\lambda_j\f$ to appear again as a spurious **ghost eigenvalue**.

**Full reorthogonalization (FRO)** (our implementation):
- Project \f$w\f$ against all \f$v_0, \ldots, v_j\f$ at each step via \f$w \leftarrow w - \sum_{l=0}^{j} (v_l^T w)\, v_l\f$
- Cost: \f$O(jn)\f$ per step, \f$O(k^2 n)\f$ total
- Guarantees orthogonality to machine precision
- Memory: must store all \f$k\f$ vectors simultaneously

---

## Performance Optimization

### BLAS-2 Reorthogonalization

MGS reorthogonalization at step \f$j\f$ performs \f$j+1\f$ BLAS-1 DOT + AXPY pairs. Replace with BLAS-2 DGEMV:

\f[h = V[:,0:j+1]^T w \quad \to \quad \texttt{dgemv}(V[:,0:j+1]^T, w) \qquad O(jn), \text{ one call}\f]
\f[w \leftarrow w - V[:,0:j+1]\, h \quad \to \quad \texttt{dgemv}(V[:,0:j+1], h) \qquad O(jn), \text{ one call}\f]

Replaces \f$j\f$ DDOT \f$+ j\f$ DAXPY with 2 DGEMV calls: \f$V[:,0:j+1]\f$ is loaded once from cache for both operations.

### Block Lanczos (BLAS-3)

Replace scalar Lanczos vectors with a block of \f$b\f$ vectors. The recurrence becomes:

\f[W = A\, V[:,j:j+b] \qquad \to \qquad \text{DGEMM: } O(n^2 b)\f]
\f[H = V[:,0:j]^T W \qquad \to \qquad \text{DGEMM: } O(jnb)\f]
\f[W \leftarrow W - V[:,0:j]\, H \qquad \to \qquad \text{DGEMM: } O(jnb)\f]
\f[[Q, R] = \text{thin QR}(W) \qquad \to \qquad \text{DGEQRF: } O(nb^2)\f]

The tridiagonal \f$T_k\f$ becomes a block tridiagonal with \f$b \times b\f$ diagonal blocks. Advantages: BLAS-3 for all major operations -> near-peak GFLOP/s; naturally resolves \f$b\f$-fold degenerate eigenvalues.

### Sparse Matrix Formats for SpMV

| Format | Memory | Access | Best for |
|--------|--------|--------|----------|
| CSR | \f$2\,\text{nnz} + n + 1\f$ | Irregular column | General sparse |
| ELLPACK | \f$\text{nnz}_{\text{padded}}\f$ | Uniform row width | Regular sparsity, GPU |
| BCSR (block) | \f$\sim \text{nnz}\f$ | Better locality | Block FEM/FD |
| Stencil (no storage) | \f$O(1)\f$ | Streaming | Structured grids |

For structured grids (finite differences), implement the matvec as a stencil loop with **no matrix storage** at all -- perfectly streaming, SIMD-friendly, and memory-optimal.

### Distributed Lanczos (MPI)

Distribute rows of \f$A\f$ and \f$V\f$ across \f$p\f$ processes. Each process holds rows \f$[i_{\text{start}}, i_{\text{end}})\f$:

```
// Local SpMV + Allreduce boundary exchange
w_local <- A_local * vj_local
MPI_Allreduce(w_local, w, SUM)         // one global reduction per matvec

// Local dot product + Allreduce
alpha_local <- vj_local^T w_local
MPI_Allreduce(&alpha_local, &alpha, SUM)
```

Communication per step: 2 Allreduce operations = \f$O(\log p)\f$ latency. Total for \f$k\f$ steps: \f$O(k \log p)\f$ latency synchronizations. The inner eigensolver on \f$T\f$ is replicated on all processes (small \f$k \times k\f$, \f$O(k^3)\f$).

For large \f$k\f$ (k > 1000), the reorthogonalization step `V[:,0:j]^T w` requires a distributed PDGEMV in ScaLAPACK or a custom blocked DGEMV across the partitioned \f$V\f$ matrix.
