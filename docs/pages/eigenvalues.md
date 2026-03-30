# Eigenvalue Algorithms {#page_eigenvalues}

Eigenvalue problems arise throughout scientific computing: structural vibration analysis,
quantum chemistry, graph clustering, principal component analysis, and more. The choice of
algorithm depends on two primary factors: whether the full decomposition is needed (all
\f$n\f$ eigenpairs) or only a few selected eigenvalues, and whether the matrix is dense or
large and sparse.

- **Full decomposition** -- Dense symmetric matrices of moderate size (\f$n \lesssim 2000\f$):
  use Cyclic Jacobi (@ref num::eig_sym) or LAPACK `dsyev`.
- **Selected eigenvalues near a target** -- Use inverse or Rayleigh quotient iteration
  (@ref num::inverse_iteration, @ref num::rayleigh_iteration).
- **A few extreme eigenvalues of a large sparse matrix** -- Use Lanczos
  (@ref num::lanczos).

---

## Cyclic Jacobi Eigendecomposition {#sec_jacobi_eig}

The Jacobi method computes the full spectral decomposition of a real symmetric matrix
\f$A = V\Lambda V^T\f$ by repeatedly applying Givens (plane) rotations that drive all
off-diagonal entries to zero.

### Similarity Transform

A Givens rotation \f$G(p,q,\theta)\f$ is the identity with a \f$2\times 2\f$ rotation
block at rows/columns \f$p\f$ and \f$q\f$. The similarity transform

\f[
  A' = G^T A G
\f]

modifies only rows and columns \f$p\f$ and \f$q\f$. Setting \f$A'_{pq} = 0\f$ requires

\f[
  (c^2 - s^2) A_{pq} + cs(A_{pp} - A_{qq}) = 0.
\f]

### Rotation Parameters

Let \f$t = \tan\theta\f$ and define

\f[
  \tau = \frac{A_{qq} - A_{pp}}{2 A_{pq}}.
\f]

Then \f$t\f$ satisfies \f$t^2 + 2\tau t - 1 = 0\f$, and taking the **smaller root**
\f$|t| \leq 1\f$ to minimize perturbation to other entries:

\f[
  t = \frac{\operatorname{sign}(\tau)}{|\tau| + \sqrt{1 + \tau^2}}, \qquad
  c = \frac{1}{\sqrt{1+t^2}}, \qquad s = ct.
\f]

When \f$\tau \gg 1\f$ this gives \f$t \approx 1/(2\tau) \approx 0\f$, avoiding catastrophic
cancellation.

### Diagonal and Off-Diagonal Update Formulas

After the rotation the updated diagonal entries are

\f[
  A'_{pp} = c^2 A_{pp} - 2cs\,A_{pq} + s^2 A_{qq}, \qquad
  A'_{qq} = s^2 A_{pp} + 2cs\,A_{pq} + c^2 A_{qq},
\f]

and the off-diagonal row/column updates for \f$r \neq p,q\f$ are

\f[
  A'_{rp} = c\,A_{rp} - s\,A_{rq}, \qquad A'_{rq} = s\,A_{rp} + c\,A_{rq}.
\f]

Eigenvectors are accumulated by applying the same rotation to the columns of \f$V\f$.

### Convergence

The off-diagonal Frobenius norm \f$\sigma_{\mathrm{off}} = \sqrt{2\sum_{p<q} A_{pq}^2}\f$
is the convergence monitor. Each rotation decreases it by exactly \f$2A_{pq}^2\f$:

\f[
  \sigma_{\mathrm{off}}^2(A') = \sigma_{\mathrm{off}}^2(A) - 2A_{pq}^2.
\f]

Over a full cyclic sweep (all \f$\binom{n}{2}\f$ pairs):

\f[
  \sigma_{\mathrm{off}}^2(A_{k+1}) \leq \left(1 - \frac{2}{n(n-1)}\right)\sigma_{\mathrm{off}}^2(A_k)
  \qquad \text{(linear phase)}.
\f]

Near the solution, convergence transitions to a **quadratic ultimate phase**:

\f[
  \sigma_{\mathrm{off}}(A_{k+1}) \leq C\,\sigma_{\mathrm{off}}^2(A_k).
\f]

In practice 5-10 sweeps suffice for double precision with well-separated eigenvalues; up to
15 sweeps for clustered spectra.

### Complexity

Each rotation costs \f$O(n)\f$ flops for the row/column update and \f$O(n)\f$ for
eigenvector accumulation. A full sweep visits \f$n(n-1)/2\f$ pairs, giving
\f$O(4n^3)\f$ flops per sweep and \f$O(n^3 \times \text{sweeps})\f$ overall.

### Comparison with Other Eigensolvers

| Method | Structure | Cost | Finds | Use when |
|--------|-----------|------|-------|----------|
| Cyclic Jacobi | Symmetric dense | \f$O(n^3 \times\f$ sweeps) | All | Small \f$n\f$, high accuracy |
| LAPACK `dsyev` (Householder+QR) | Symmetric dense | \f$O(\tfrac{4}{3}n^3)\f$ | All | Standard dense symmetric |
| Power iteration | Any | \f$O(n^2)\times\f$ iters | 1 dominant | Sparse, dominant only |
| Lanczos | Symmetric sparse | \f$O(kn)\times\f$ iters | \f$k\f$ extreme | Large sparse, \f$k \ll n\f$ |

**API**: @ref num::eig_sym, @ref num::EigenResult

---

## Power, Inverse, and Rayleigh Iteration {#sec_power}

These single-vector methods find **one** eigenpair at a time and are most useful when only
one or a few eigenvalues are needed.

### Power Iteration

Expand the starting vector in the eigenbasis: \f$v_0 = \sum_i \alpha_i u_i\f$. After
\f$k\f$ applications of \f$A\f$, the component along \f$u_i\f$ (\f$i \geq 2\f$) decays
as \f$(\lambda_i/\lambda_1)^k\f$, so the iteration converges to the dominant eigenvector.

The update rule is

\f[
  \mathbf{v}_{k+1} = \frac{A\mathbf{v}_k}{\|A\mathbf{v}_k\|}.
\f]

The eigenvalue estimate at each step is the Rayleigh quotient \f$\lambda \approx \mathbf{v}_k^T A \mathbf{v}_k\f$.

**Convergence rate**: the angle between \f$v_k\f$ and the true eigenvector \f$u_1\f$ decays as

\f[
  \left|\frac{\lambda_2}{\lambda_1}\right|^k.
\f]

When \f$|\lambda_1| \approx |\lambda_2|\f$ convergence is very slow; use Lanczos instead.

**API**: @ref num::power_iteration

### Inverse Iteration

Replace \f$A\f$ by \f$(A - \sigma I)^{-1}\f$. Its eigenvalues are \f$\{1/(\lambda_i - \sigma)\}\f$,
so the dominant eigenvalue of the shifted inverse corresponds to the \f$\lambda_i\f$
**closest to \f$\sigma\f$**. Factor \f$(A - \sigma I)\f$ once (\f$O(n^3)\f$), then solve
cheaply (\f$O(n^2)\f$) at each iteration.

The convergence factor at shift \f$\sigma\f$ is

\f[
  r = \frac{|\lambda - \sigma|}{|\mu - \sigma|}
\f]

where \f$\lambda\f$ is the nearest eigenvalue and \f$\mu\f$ is the second nearest. Choosing
\f$\sigma\f$ close to the target makes \f$r \approx 0\f$; typically 3-5 iterations reach
machine precision from a shift within 10% of \f$\lambda\f$.

**API**: @ref num::inverse_iteration

### Rayleigh Quotient Iteration

Update the shift at every step using the Rayleigh quotient of the current vector:

\f[
  \sigma_k = \mathbf{v}_k^T A \mathbf{v}_k \qquad (\|\mathbf{v}_k\| = 1).
\f]

For symmetric \f$A\f$ this gives **cubic convergence**: if the current error is
\f$\epsilon\f$, then

\f[
  |\sigma_{k+1} - \lambda| \leq C\,|\sigma_k - \lambda|^3.
\f]

An error of \f$10^{-4}\f$ becomes \f$\sim 10^{-12}\f$ in one more step. The cost is
\f$O(n^3)\f$ per iteration (new LU at each step), so in practice RQI is used only as a
cheap refinement pass (<= 3 iterations) after a rough estimate from power or inverse
iteration.

**API**: @ref num::rayleigh_iteration

### Method Selection

| Goal | Recommended method |
|------|--------------------|
| Largest eigenvalue, well-separated | @ref num::power_iteration |
| Eigenvalue nearest known \f$\sigma\f$ | @ref num::inverse_iteration |
| Refine a rough estimate to machine precision | @ref num::rayleigh_iteration |
| Several extreme eigenvalues, large sparse \f$A\f$ | @ref num::lanczos |

---

## Lanczos Algorithm {#sec_lanczos}

Lanczos builds a \f$k\f$-dimensional Krylov subspace
\f$\mathcal{K}_k(A, v_1) = \operatorname{span}\{v_1, Av_1, \ldots, A^{k-1}v_1\}\f$
and extracts near-optimal approximations to the extreme eigenvalues of a large symmetric
matrix. It is the method of choice when \f$k \ll n\f$ and \f$A\f$ is available only as a
matrix-vector product (SpMV).

### The Lanczos Relation

Starting from a unit vector \f$v_1\f$, the \f$k\f$-step process constructs an orthonormal
basis \f$V_k = [v_1, \ldots, v_k]\f$ satisfying

\f[
  AV_k = V_k T_k + \beta_k \mathbf{v}_{k+1} \mathbf{e}_k^T,
\f]

where \f$T_k\f$ is the symmetric **tridiagonal** Galerkin projection

\f[
  T_k = V_k^T A V_k \in \mathbb{R}^{k \times k}.
\f]

The three-term recurrence generating each new basis vector is:
\f$w = Av_j\f$, \f$\alpha_j = v_j^T w\f$, \f$w \leftarrow w - \alpha_j v_j - \beta_{j-1} v_{j-1}\f$,
\f$\beta_j = \|w\|\f$, \f$v_{j+1} = w/\beta_j\f$.

### Ritz Values and Cheap Residual Bound

The eigenvalues of \f$T_k\f$ -- called **Ritz values** \f$\theta_1, \ldots, \theta_k\f$ --
approximate the extreme eigenvalues of \f$A\f$. If \f$S\f$ is the eigenvector matrix of
\f$T_k\f$, the Ritz vectors \f$\tilde{u}_i = V_k s_i\f$ satisfy

\f[
  \|A\tilde{u}_i - \theta_i \tilde{u}_i\|_2 = \beta_k |S_{k,i}|.
\f]

This provides an inexpensive stopping criterion: only \f$\beta_k\f$ (already computed in
the recurrence) and the last row of \f$S\f$ (from the small eigenproblem on \f$T_k\f$) are
needed -- no explicit Ritz vector is required.

### Ghost Eigenvalues and Full Reorthogonalization

In floating-point arithmetic, the three-term recurrence loses orthogonality after
\f$O(1/\varepsilon_{\mathrm{mach}})\f$ steps. Once a Ritz value converges to an
eigenvalue \f$\lambda_j\f$, the corresponding Lanczos vector drifts back into the
\f$u_j\f$ eigenspace, causing \f$\lambda_j\f$ to reappear as a spurious **ghost
eigenvalue**.

**Full reorthogonalization (MGS)** projects \f$w\f$ against all previous basis vectors at
every step, at a cost of \f$O(k^2 n)\f$ total, and guarantees orthogonality to machine
precision. It requires storing all \f$k\f$ vectors simultaneously.

Selective reorthogonalization (Simon 1984) reorthogonalizes only against converged Ritz
vectors, reducing work by 2-10x when few eigenvalues have converged.

### Thick Restart and ARPACK

**Thick restart** (Wu-Simon 2000) retains the \f$p\f$ best Ritz pairs when restarting,
forming the new starting basis \f$[\tilde{u}_1, \ldots, \tilde{u}_p, v_{k+1}]\f$. This is
the basis of **ARPACK** (Implicitly Restarted Lanczos), which is the backend for MATLAB
`eigs()` and SciPy `eigsh()`.

### Complexity

| Phase | Cost |
|-------|------|
| \f$k\f$ matvecs | \f$O(k \cdot \mathrm{nnz})\f$ sparse, \f$O(kn^2)\f$ dense |
| Full reorthogonalization | \f$O(k^2 n)\f$ |
| Inner eigensolver on \f$T_k\f$ | \f$O(k^3)\f$ (negligible for \f$k \ll n\f$) |

**API**: @ref num::lanczos, @ref num::LanczosResult
