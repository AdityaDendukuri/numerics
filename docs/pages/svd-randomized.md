# Randomized Truncated SVD {#page_svd_random_notes}

## Problem Statement

Given \f$A \in \mathbb{R}^{m \times n}\f$ and a target rank \f$k \ll \min(m, n)\f$, find an approximate rank-\f$k\f$ SVD:

\f[A \approx \hat{U}_k \hat{\Sigma}_k \hat{V}_k^T, \qquad \hat{U}_k \in \mathbb{R}^{m \times k},\ \hat{\Sigma}_k \in \mathbb{R}^{k \times k},\ \hat{V}_k \in \mathbb{R}^{n \times k}\f]

minimizing \f$\|A - \hat{U}_k \hat{\Sigma}_k \hat{V}_k^T\|_F\f$ subject to rank \f$\leq k\f$.

The **Eckart-Young theorem** guarantees that the exact rank-\f$k\f$ truncation (using the \f$k\f$ largest singular values) is the best rank-\f$k\f$ approximation. The randomized SVD approximates this optimum cheaply when \f$\sigma_{k+1} \ll \sigma_k\f$ (fast spectral decay).

**Reference**: Halko, Martinsson, Tropp, "Finding Structure with Randomness" (SIAM Review 2011).

---

## Key Insight: Randomized Range Finding

The column space of \f$A\f$ is approximated by a small random sketch. If \f$\Omega \in \mathbb{R}^{n \times \ell}\f$ is a Gaussian random matrix (\f$\ell = k + p\f$, \f$p\f$ = oversampling), then:

\f[Y = A\Omega \in \mathbb{R}^{m \times \ell}\f]

has range \f$\approx \operatorname{range}(A)\f$ with high probability. A QR factorization \f$Y = Q\hat{R}\f$ gives an orthonormal basis \f$Q\f$ for this approximate range.

The random sketch mixes right singular vectors with coefficients proportional to \f$\sigma_i\f$, so large singular values dominate and the range of \f$Y\f$ concentrates near the top-\f$k\f$ singular subspace with high probability.

**Error bound** (Halko-Martinsson-Tropp): for Gaussian \f$\Omega\f$ with \f$\ell = k + p\f$, \f$p \geq 2\f$:

\f[\mathbb{E}\|A - QQ^TA\|_F \leq \left(1 + \frac{k}{p-1}\right)^{1/2} \left(\sum_{j > k} \sigma_j^2\right)^{1/2}\f]

The right-hand side equals the optimal tail error \f$\left(\sum_{j>k}\sigma_j^2\right)^{1/2}\f$ times a factor close to 1 for \f$p \geq k\f$. In practice \f$p = 10\f$ suffices.

---

## Algorithm

```
function svd_truncated(A, k, p, rng):
    m, n <- A.rows, A.cols
    l    <- min(k + p, min(m, n))   // sketch size (capped at matrix rank bound)

    // Step 1: Gaussian random sketch Omega in R^{n x l}
    Omega[i,j] ~ N(0, 1)  iid

    // Step 2: Y = A * Omega  in R^{m x l}
    Y <- A * Omega                   // DGEMM: O(mnl)

    // Step 3: Q = thin QR(Y) in R^{m x l}, orthonormal columns
    [Q, ~] <- thin_qr(Y)            // O(ml^2)

    // Step 4: B = Q^T * A  in R^{l x n}
    B <- Q^T * A                     // DGEMM: O(mnl)

    // Step 5: Full SVD of the small B in R^{l x n}
    [U_B, Sigma, Vt] <- svd(B)      // O(l^2 n) or O(ln^2)

    // Step 6: U = Q * U_B[:,0:k], keep top-k
    U <- Q * U_B[:,0:k]             // DGEMM: O(mlk)

    return (U[:,0:k], Sigma[0:k], Vt[0:k, :])
```

**Total cost**: \f$O(mn\ell)\f$ for the two DGEMM calls + \f$O(m\ell^2)\f$ for QR + \f$O(\ell^2 n)\f$ for the small SVD.

For \f$\ell = k + p \ll \min(m, n)\f$: **dramatically cheaper** than exact SVD at \f$O(mn\min(m,n))\f$.

---

## Step-by-Step Analysis

### Step 1: Random Sketch

The sketch \f$\Omega \sim \mathcal{N}(0,1)^{n \times \ell}\f$ acts as a "sampler" of the column space. Alternative random matrices with similar guarantees:

- **Subsampled Randomized Hadamard Transform (SRHT)**: \f$\Omega = \sqrt{n/\ell}\, D H S^T\f$ where \f$D\f$ is a diagonal \f$\pm 1\f$ matrix, \f$H\f$ is the Walsh-Hadamard transform, and \f$S\f$ selects \f$\ell\f$ random columns. Cost: \f$O(n \log n)\f$ per column instead of \f$O(n\ell)\f$ for Gaussian -- better for very large \f$n\f$.
- **Sparse random maps** (CountSketch): \f$\Omega\f$ has one \f$\pm 1\f$ per column, all others zero. Cost: \f$O(\text{nnz}(A))\f$ for the sketch, works for sparse \f$A\f$.
- **Structured random maps**: FJLT (Fast Johnson-Lindenstrauss) -- \f$O(n \log \ell)\f$ per column.

### Step 2: Matrix-Vector Product \f$Y = A\Omega\f$

This is a DGEMM: \f$m \times n\f$ times \f$n \times \ell\f$ -> \f$m \times \ell\f$. Cost: \f$2mn\ell\f$ flops.

For very large \f$A\f$ (doesn't fit in RAM): compute \f$Y\f$ in row blocks, accumulating columns of \f$Y\f$ from partial sums:

\f[Y = \sum_{b=1}^{B} A_b \Omega_b, \qquad A_b \in \mathbb{R}^{m_b \times n},\ \Omega_b = \Omega[\text{rows of }b, :]\f]

Each block \f$A_b\f$ is loaded from disk once, \f$\Omega_b\f$ fits in RAM -- only one pass over \f$A\f$ required.

### Step 3: QR of \f$Y\f$

Thin QR of \f$m \times \ell\f$ matrix: \f$O(m\ell^2)\f$ flops. For \f$\ell \ll m\f$: cheap relative to the DGEMM.

**Power iteration** (optional, for slowly decaying spectra): replace \f$Y = A\Omega\f$ with \f$Y = (AA^T)^q A\Omega\f$ for some \f$q \geq 1\f$:

\f[Y \leftarrow A\Omega, \quad \text{then repeat } q \text{ times:} \quad Y \leftarrow A(A^T Y)\f]

Each power step re-orthogonalizes \f$Y\f$ via QR and costs \f$2 \times O(mn\ell)\f$ extra flops. The effective singular value decay is raised to \f$\sigma_j^{2q+1}\f$, dramatically improving approximation quality for slowly decaying spectra. Use \f$q = 1\f$--\f$3\f$ for images or text data.

### Step 4: Projection \f$B = Q^TA\f$

Another DGEMM: \f$\ell \times m\f$ times \f$m \times n\f$ -> \f$\ell \times n\f$. Cost: \f$2mn\ell\f$ flops. Together with Step 2, these two DGEMMs dominate the total cost.

### Step 5: Small SVD of \f$B\f$

\f$B \in \mathbb{R}^{\ell \times n}\f$ with \f$\ell = k + p \ll n\f$: standard Jacobi SVD or `dgesdd`. Cost: \f$O(\ell^2 n)\f$ (thin case, \f$\ell < n\f$) or \f$O(\ell n^2)\f$ (fat case) -- small in either case.

### Step 6: Lift to Full Space \f$U = Q\hat{U}_\ell\f$

DGEMM: \f$m \times \ell\f$ times \f$\ell \times k\f$ -> \f$m \times k\f$. Cost: \f$2m\ell k\f$.

---

## Oversampling Cap: Why \f$\ell \leq \min(m,n)\f$

If \f$\ell > \min(m, n)\f$: the sketch \f$Y = A\Omega \in \mathbb{R}^{m \times \ell}\f$ has rank at most \f$\min(m,n)\f$. The thin QR of \f$Y\f$ produces at most \f$m\f$ orthonormal columns. Attempting to use \f$\ell > m\f$ columns from a rank-\f$m\f$ QR causes out-of-bounds access.

**Fix** (implemented in our code): cap \f$\ell \leftarrow \min(k + p, \min(m, n))\f$ before proceeding.

---

## Accuracy vs Cost Trade-off

| Parameter | Effect on accuracy | Effect on cost |
|-----------|--------------------|----------------|
| Oversampling \f$p\f$ | \f$+\f$ improves (error factor \f$\to 1\f$ as \f$p \to \infty\f$) | \f$+\f$ increases (linearly in \f$p\f$) |
| Power iterations \f$q\f$ | \f$+\f$ dramatically improves for slow decay | \f$+\f$ increases (\f$2q\f$ extra DGEMM passes) |
| Sketch type (Gaussian vs SRHT) | similar guarantees | Gaussian: \f$O(mn\ell)\f$; SRHT: \f$O(mn\log\ell)\f$ |

**Rule of thumb** (Halko et al.):
- \f$p = 5\f$-\f$10\f$ for well-separated spectra (image compression, PCA on clean data)
- \f$q = 1\f$-\f$3\f$ power iterations for slowly decaying spectra (recommendation systems, text data)

---

## Applications

| Application | \f$m\f$ | \f$n\f$ | \f$k\f$ | Notes |
|---|---|---|---|---|
| Image compression | \f$10^6\f$ | \f$10^6\f$ | 50 | Fast singular values for JPEG-SVD |
| PCA / dimensionality reduction | \f$10^5\f$ | \f$10^4\f$ | 100 | \f$\Omega\f$ can be structured (SRHT) |
| Latent semantic analysis (text) | \f$10^6\f$ | \f$10^5\f$ | 300 | Slow decay -> need \f$q \geq 1\f$ |
| Low-rank matrix completion | varies | varies | varies | Used in Netflix prize algorithms |
| Physics preconditioning | \f$n\f$ | \f$n\f$ | \f$k \ll n\f$ | Approximate spectral preconditioner |

---

## Performance Optimization

### Current Implementation

Two DGEMM calls (Steps 2 and 4) plus QR and small SVD. The DGEMM calls dominate for large \f$m, n\f$.

**BLAS optimization** (already in use via `Backend::blas`): `cblas_dgemm` routes to AVX2/AVX-512 microkernel. For \f$m = n = 10^4\f$, \f$\ell = 50\f$: DGEMM achieves ~400 GFLOP/s vs ~0.5 GFLOP/s naive.

### Streaming Computation (Out-of-Core)

For \f$A\f$ too large for RAM (common in data science):

```
// Streaming Y = A * Omega in row blocks
Y <- 0 (m x l)
for each row-block A_b of A from disk:
    Y += A_b * Omega[rows of b, :]     // one DGEMM per block
// Then proceed: QR(Y), B = Q^T * A (second streaming pass), etc.
```

Cost: 2 passes over \f$A\f$ (once for \f$Y\f$, once for \f$B\f$). Memory: \f$O(m\ell + n\ell)\f$ for \f$Q\f$ and \f$\Omega\f$.

### GPU Implementation

Both DGEMM calls in Steps 2 and 4 are ideal for GPU:

```
// On GPU: cuBLAS cublasDgemm for Steps 2 and 4
// Gaussian sketch: cuRAND for fast generation of Omega
// QR: cuSOLVER cusolverDnDgeqrf (blocked Householder on GPU)
// Small SVD: host (l << n, cheap)
```

For \f$m = n = 10^4\f$, \f$\ell = 100\f$: GPU DGEMM at 10 TFLOP/s -> Steps 2+4 in \f${\sim}2\f$ ms total. Compare with exact SVD (`dgesdd`): \f$O(mn^2) \sim 10^{12}\f$ flops -> 1000 s on CPU, 100 s on GPU.

### Avoiding Step 4 via Single-View Algorithm

If \f$A\f$ is available only in a single pass (streaming):

```
// Single-view randomized SVD (Tropp et al. 2017)
// Generate two sketches simultaneously:
Y = A * Omega         // right sketch (m x l)
Z = Psi^T * A         // left sketch (l x n),  Psi in R^{m x l}

// Recover B = Q^T A from Y and Z without a second pass over A:
Q, R_Y <- QR(Y)
B <- solve (R_Y^T B = Z) for B    // O(l^2 n)
```

This uses only one pass over \f$A\f$, at the cost of slightly worse accuracy. Needed for sensor data streams or when re-reading \f$A\f$ is expensive.

### Structured Sketches for Sparse \f$A\f$

For sparse \f$A\f$ with \f$\text{nnz} \ll mn\f$: use a **CountSketch** (one non-zero per column of \f$\Omega\f$):

\f[Y_{ij} \leftarrow Y_{h(j), :} \mathrel{+}= \eta_j \cdot A[:, j], \quad h(j) \in \{0,\ldots,\ell-1\}, \quad \eta_j \in \{+1,-1\}\f]

Cost: \f$O(\text{nnz}(A))\f$ -- same as one SpMV. The sketch sacrifices some accuracy (bounded by \f$O(1/\sqrt{\ell})\f$ error) for extreme speed.
