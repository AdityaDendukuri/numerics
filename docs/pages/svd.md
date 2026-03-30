# Singular Value Decomposition {#page_svd}

Every real matrix \f$A \in \mathbb{R}^{m \times n}\f$ admits the **singular value
decomposition**

\f[
  A = U \Sigma V^T,
\f]

where \f$U \in \mathbb{R}^{m \times r}\f$ and \f$V \in \mathbb{R}^{n \times r}\f$ have
orthonormal columns, \f$r = \min(m,n)\f$, and
\f$\Sigma = \operatorname{diag}(\sigma_1, \ldots, \sigma_r)\f$ with
\f$\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r \geq 0\f$ are the **singular
values**. The *economy* (thin) form stores only \f$r\f$ columns of \f$U\f$ and \f$V\f$;
the *full* form pads \f$U\f$ or \f$V\f$ to square. This library returns the economy form.

Two complementary algorithms are provided:

- **One-sided Jacobi SVD** (@ref num::svd) -- dense, high relative accuracy for all
  singular values, including those close to zero.
- **Randomized truncated SVD** (@ref num::svd_truncated) -- finds the \f$k\f$ largest
  singular values cheaply when \f$k \ll \min(m,n)\f$ and fast spectral decay holds.

---

## One-Sided Jacobi SVD {#sec_svd_jacobi}

The one-sided Jacobi method applies Givens rotations **from the right** to the columns of
\f$A\f$ until they are mutually orthogonal. At convergence, column norms equal the
singular values and normalized columns equal the left singular vectors.

### Why Not Form \f$A^T A\f$

The singular values of \f$A\f$ are the square roots of the eigenvalues of \f$A^T A\f$.
Explicitly forming \f$A^T A\f$ squares the condition number:
\f$\kappa(A^T A) = \kappa(A)^2\f$. For ill-conditioned \f$A\f$ this causes severe loss of
accuracy in the smallest singular values. The one-sided method works directly on \f$A\f$
and avoids this entirely.

### Column Orthogonality Condition

For column pair \f$(p, q)\f$, define the inner products

\f[
  \alpha = \|a_p\|^2, \qquad \beta = \|a_q\|^2, \qquad \gamma = a_p \cdot a_q.
\f]

The relative cosine \f$|\gamma|/\sqrt{\alpha\beta}\f$ measures deviation from
orthogonality. A rotation is skipped whenever this quantity is below the convergence
tolerance.

### Rotation Angle

To zero \f$[A^T A]_{pq} = \gamma\f$ via a right rotation \f$A \leftarrow AG\f$, define

\f[
  \zeta = \frac{\beta - \alpha}{2\gamma}.
\f]

Then \f$t\f$ is the smaller root of \f$t^2 + 2\zeta t - 1 = 0\f$:

\f[
  t = \frac{\operatorname{sign}(\zeta)}{|\zeta| + \sqrt{1 + \zeta^2}}, \qquad
  c = \frac{1}{\sqrt{1+t^2}}, \qquad s = ct.
\f]

With \f$a_p' = ca_p - sa_q\f$ and \f$a_q' = sa_p + ca_q\f$, one can verify that
\f$a_p'^T a_q' = 0\f$ by construction.

### Convergence

Define the **off-orthogonality** functional

\f[
  \psi(A) = \sum_{p < q} \gamma_{pq}^2.
\f]

Each rotation decreases \f$\psi\f$ by exactly \f$\gamma_{pq}^2\f$:

\f[
  \psi(A') = \psi(A) - \gamma_{pq}^2.
\f]

After a full sweep over all \f$\binom{r}{2}\f$ pairs:

\f[
  \psi(A_{\mathrm{sweep}+1}) \leq \left(1 - \frac{2}{r(r-1)}\right)\psi(A_{\mathrm{sweep}})
  \qquad \text{(linear phase)}.
\f]

Once \f$\max_{p \neq q}|[A^T A]_{pq}|\f$ is small the method enters a quadratic phase,
identical to the analysis of symmetric Jacobi applied to \f$A^T A\f$. In practice 5-10
sweeps suffice for double precision.

### High Relative Accuracy

One-sided Jacobi is among the most accurate SVD algorithms. Whereas standard
bidiagonalization (LAPACK `dgesdd`) computes all singular values to absolute accuracy
\f$O(\varepsilon_{\mathrm{mach}}\|A\|)\f$, the Jacobi method achieves high **relative**
accuracy for small singular values: if \f$A\f$ is column-equilibrated,
\f$\sigma_k\f$ is computed with relative error \f$O(\varepsilon_{\mathrm{mach}})\f$ even
when \f$\sigma_k / \sigma_1 = O(\varepsilon_{\mathrm{mach}})\f$.

### Complexity and Comparison

Complexity per sweep: \f$O(r^2)\f$ rotations \f$\times\f$ \f$O(m + n)\f$ flops per
rotation \f$= O(r^2(m+n))\f$.

| Algorithm | Cost | Accuracy | Use when |
|-----------|------|----------|----------|
| One-sided Jacobi (this library) | \f$O(r^2(m+n)\times\f$ sweeps) | High relative | Small \f$n\f$, need small \f$\sigma_k\f$ accurately |
| LAPACK `dgesvd` (bidiag + QR) | \f$O(mn^2)\f$ | Absolute | Standard dense SVD |
| LAPACK `dgesdd` (divide & conquer) | \f$O(mn^2)\f$ | Absolute | Fast, most common |
| Randomized SVD | \f$O(mn\ell)\f$ | Approximate | \f$k \ll \min(m,n)\f$ |

For tall-skinny \f$A\f$ (\f$m \gg n\f$): compute thin QR first (\f$A = \hat{Q}\hat{R}\f$),
run Jacobi on \f$\hat{R} \in \mathbb{R}^{n \times n}\f$, then recover
\f$U = \hat{Q}\hat{U}\f$.

**API**: @ref num::svd, @ref num::SVDResult

---

## Randomized Truncated SVD {#sec_svd_randomized}

When only the \f$k\f$ largest singular values are needed and \f$k \ll \min(m,n)\f$, a full
SVD wastes almost all its work. The randomized algorithm exploits random sketching to
capture the dominant singular subspace cheaply.

### Eckart-Young Theorem

The best rank-\f$k\f$ approximation to \f$A\f$ in both the Frobenius and spectral norms is
the truncated SVD using the \f$k\f$ largest singular values:

\f[
  \min_{\operatorname{rank}(B) \leq k} \|A - B\|_F = \left(\sum_{j > k} \sigma_j^2\right)^{1/2}.
\f]

The randomized algorithm approximates this optimum with a controllable error that vanishes
as oversampling increases.

### Algorithm (Halko-Martinsson-Tropp 2011)

Given target rank \f$k\f$ and oversampling \f$p\f$ (typically \f$p = 5\f$-\f$10\f$), set
\f$\ell = k + p\f$ capped at \f$\min(m, n)\f$ to avoid rank overflow.

1. Draw a Gaussian sketch \f$\Omega \in \mathbb{R}^{n \times \ell}\f$ with
   \f$\Omega_{ij} \sim \mathcal{N}(0,1)\f$.
2. Form \f$Y = A\Omega \in \mathbb{R}^{m \times \ell}\f$ (one DGEMM).
3. Thin QR: compute orthonormal \f$Q \in \mathbb{R}^{m \times \ell}\f$ such that
   \f$\operatorname{range}(Q) \approx \operatorname{range}(A)\f$.
4. Project: \f$B = Q^T A \in \mathbb{R}^{\ell \times n}\f$ (one DGEMM).
5. Compute the full SVD of the small matrix \f$B\f$: \f$B = \tilde{U}\Sigma V^T\f$.
6. Lift to the original space: \f$U = Q\tilde{U}_{[:,\,0:k]}\f$.

Return \f$(U_{[:,0:k]},\,\Sigma_{0:k},\,V^T_{0:k,:})\f$.

### HMT Error Bound

For Gaussian \f$\Omega\f$ with \f$\ell = k + p\f$, \f$p \geq 2\f$:

\f[
  \mathbb{E}\|A - QQ^T A\|_F
    \leq \left(1 + \frac{k}{p-1}\right)^{1/2}
         \left(\sum_{j > k} \sigma_j^2\right)^{1/2}.
\f]

The factor in front of the optimal tail error approaches 1 as \f$p \to \infty\f$. In
practice \f$p = 10\f$ is sufficient for well-separated spectra.

### Oversampling Cap \f$\ell \leq \min(m,n)\f$

If \f$\ell > \min(m,n)\f$, the sketch \f$Y\f$ has rank at most \f$\min(m,n)\f$ and the
thin QR produces at most \f$m\f$ orthonormal columns. Requesting more columns causes
out-of-bounds access, so the implementation clamps
\f$\ell \leftarrow \min(k+p,\,\min(m,n))\f$ before proceeding.

### Accuracy vs Cost Trade-offs

| Parameter | Effect on accuracy | Effect on cost |
|-----------|-------------------|----------------|
| Oversampling \f$p\f$ | Increases (error factor \f$\to 1\f$ as \f$p \to \infty\f$) | Linear in \f$p\f$ |
| Power iterations \f$q\f$ | Dramatic improvement for slow spectral decay | \f$2q\f$ extra DGEMM passes |
| Gaussian vs SRHT sketch | Similar guarantees | SRHT: \f$O(mn\log\ell)\f$ vs Gaussian: \f$O(mn\ell)\f$ |

Optional power iterations replace \f$Y = A\Omega\f$ with
\f$Y = (AA^T)^q A\Omega\f$, raising singular value decay to the \f$q\f$-th power. This
dramatically improves approximation quality for slowly decaying spectra such as text or
recommendation-system matrices; \f$q = 1\f$-\f$3\f$ typically suffices.

**Total cost**: \f$O(mn\ell)\f$ (two DGEMMs) \f$+ O(m\ell^2)\f$ (thin QR)
\f$+ O(\ell^2 n)\f$ (small SVD). For \f$\ell \ll \min(m,n)\f$ this is dramatically
cheaper than the exact \f$O(mn\min(m,n))\f$ SVD.

**API**: @ref num::svd_truncated
