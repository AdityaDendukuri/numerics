# Linear Solvers {#page_linear_solvers}

**Direct solvers** (LU, Thomas) factor the matrix once and solve in \f$O(n^2)\f$
per right-hand side; they are exact up to floating-point and preferred when
\f$n\f$ is moderate or many right-hand sides share the same coefficient matrix.
**Iterative solvers** (Jacobi, Gauss-Seidel, CG, GMRES) never form a
factorization; they produce a sequence of approximate solutions and are
preferred for large sparse systems where factorization fill-in would be
prohibitive.

---

## Thomas Algorithm (Tridiagonal) {#sec_thomas}

### The Tridiagonal System

Solve \f$Tx = d\f$ where

\f[
T = \begin{pmatrix}
b_0 & c_0 & & \\
a_1 & b_1 & c_1 & \\
& a_2 & b_2 & \ddots \\
& & \ddots & \ddots & c_{n-2} \\
& & & a_{n-1} & b_{n-1}
\end{pmatrix}
\f]

Tridiagonal systems arise in 1D finite differences, implicit time integration
(Crank-Nicolson), cubic spline interpolation, and ADI splitting for 2D/3D
parabolic equations.

### LU Factorization View

The Thomas algorithm is Gaussian elimination specialized to tridiagonal
structure.  The LU factors have the form

\f[
L = \begin{pmatrix} 1 \\ l_0 & 1 \\ & l_1 & \ddots \\ & & \ddots & 1 \end{pmatrix},
\quad
U = \begin{pmatrix} u_0 & c_0 \\ & u_1 & c_1 \\ & & \ddots & \ddots \\ & & & u_{n-1} \end{pmatrix}
\f]

where the recurrences follow directly from \f$A = LU\f$:

\f[
l_{i-1} = \frac{a_{i-1}}{u_{i-1}}, \qquad u_i = b_i - l_{i-1}\, c_{i-1}
\f]

### Forward Sweep and Back Substitution

**Forward sweep** (eliminate sub-diagonal, modify \f$d\f$ in place):
\f[
l \leftarrow a_i / u_{i-1}, \quad
u_i \leftarrow b_i - l\, c_{i-1}, \quad
d_i \leftarrow d_i - l\, d_{i-1},
\quad i = 1, \ldots, n-1
\f]

**Backward substitution**:
\f[
x_{n-1} = d_{n-1} / u_{n-1}, \qquad
x_i = (d_i - c_i\, x_{i+1}) / u_i, \quad i = n-2, \ldots, 0
\f]

### Complexity and Stability

The grand total is \f$8n - 7\f$ FLOPs -- optimal, since reading the \f$O(n)\f$
input already costs \f$O(n)\f$.

The algorithm is unconditionally stable when \f$T\f$ is **strictly diagonally
dominant** (\f$|b_i| > |a_i| + |c_i|\f$ for all \f$i\f$) or **symmetric positive
definite**.  Under diagonal dominance the modified diagonal \f$|u_i| > |c_i|\f$
remains bounded away from zero at every step.

### GPU Batched Variant

The forward sweep is inherently sequential within a single system.
For \f$m\f$ independent tridiagonal systems (common in ADI methods), assign
one GPU thread per system: the \f$m\f$ sweeps are fully independent and
the kernel achieves near-peak memory bandwidth.

### API

- @ref num::thomas -- solve a single or batched tridiagonal system

---

## Stationary Iterative Methods {#sec_stationary}

### Splitting Framework

Write \f$A = M - N\f$ and iterate

\f[
M x_{k+1} = N x_k + b \quad \Longrightarrow \quad x_{k+1} = \underbrace{M^{-1}N}_{B}\, x_k + M^{-1}b
\f]

The iteration converges from any starting point if and only if the
**spectral radius** \f$\rho(B) < 1\f$.  Decompose \f$A = D + L + U\f$ (diagonal,
strict lower, strict upper):

| Method | \f$M\f$ | \f$N\f$ |
|--------|---------|---------|
| Jacobi | \f$D\f$ | \f$-(L+U)\f$ |
| Gauss-Seidel | \f$D+L\f$ | \f$-U\f$ |
| SOR(\f$\omega\f$) | \f$\frac{1}{\omega}D + L\f$ | \f$(\frac{1}{\omega}-1)D - U\f$ |

### Jacobi Iteration

Each component of the new iterate is computed from the **old** iterate only:

\f[
x_i^{(k+1)} = \frac{1}{A_{ii}}\!\left(b_i - \sum_{j \neq i} A_{ij}\, x_j^{(k)}\right),
\quad i = 0, \ldots, n-1
\f]

Because all \f$x_i^{(k+1)}\f$ depend solely on \f$x^{(k)}\f$, Jacobi is
**embarrassingly parallel** -- all updates are independent.

**Convergence condition**: Jacobi converges under strict diagonal dominance
(\f$|A_{ii}| > \sum_{j \neq i}|A_{ij}|\f$) or for SPD \f$A\f$.  The iteration
matrix is \f$B_J = -D^{-1}(L+U)\f$.  For the model Poisson problem
(5-point stencil on an \f$n \times n\f$ grid):

\f[
\rho(B_J) = \cos\!\left(\frac{\pi}{n+1}\right) \approx 1 - \frac{\pi^2}{2n^2}
\f]

This requires \f$O(n^2)\f$ iterations to reduce the error by \f$1/e\f$ -- Jacobi
is too slow as a standalone solver for large Poisson problems but remains
a useful multigrid smoother.

- @ref num::jacobi -- Jacobi iteration for dense or sparse \f$A\f$

### Gauss-Seidel and SOR

Each component uses the **most recently computed** values:

\f[
x_i^{(k+1)} = \frac{1}{A_{ii}}\!\left(b_i
  - \sum_{j < i} A_{ij}\, x_j^{(k+1)}
  - \sum_{j > i} A_{ij}\, x_j^{(k)}\right)
\f]

The in-place update propagates fresh data immediately.  For SPD \f$A\f$ the
**Stein-Rosenberg theorem** gives

\f[
\rho(B_{GS}) = \rho(B_J)^2
\f]

so Gauss-Seidel converges in half as many iterations as Jacobi.

**SOR** over-relaxes the Gauss-Seidel correction by \f$\omega \in (0,2)\f$:

\f[
x_i^{(k+1)} = x_i^{(k)} + \omega\!\left(\tilde{x}_i^{(GS)} - x_i^{(k)}\right)
\f]

For the model Poisson problem the optimal parameter is

\f[
\omega_{\mathrm{opt}} = \frac{2}{1 + \sin(\pi/(n+1))} \approx 2 - \frac{2\pi}{n}
\f]

which reduces the spectral radius to

\f[
\rho(B_{SOR,\,\omega_{\mathrm{opt}}}) \approx 1 - \frac{4\pi}{n}
\f]

requiring only \f$O(n)\f$ iterations -- a landmark improvement over the
\f$O(n^2)\f$ of plain Gauss-Seidel (D.M. Young, 1950).

**Parallelism**: natural-ordering GS has a sequential dependency chain.
For structured grids, **red-black ordering** (color node \f$(i,j)\f$ by
parity of \f$i+j\f$) decouples all red updates and all black updates into
two independent parallel sweeps, preserving the same convergence rate.

- @ref num::gauss_seidel -- Gauss-Seidel and SOR iteration

---

## Conjugate Gradient {#sec_cg}

### Minimization Perspective

For an SPD matrix \f$A\f$, solving \f$Ax = b\f$ is equivalent to minimizing the
quadratic functional

\f[
\phi(\mathbf{x}) = \frac{1}{2}\mathbf{x}^T A\mathbf{x} - \mathbf{b}^T\mathbf{x}
\f]

since \f$\nabla\phi = A\mathbf{x} - \mathbf{b} = -\mathbf{r}\f$.  The residual
points toward the minimum; CG follows a sequence of search directions that
span the **Krylov subspace**

\f[
\mathcal{K}_k(A,\mathbf{r}_0) = \operatorname{span}\!\bigl\{\mathbf{r}_0,\, A\mathbf{r}_0,\, \ldots,\, A^{k-1}\mathbf{r}_0\bigr\}
\f]

The iterate \f$x_k\f$ is the unique minimizer of \f$\|\mathbf{x}-\mathbf{x}^*\|_A\f$
over \f$\mathbf{x}_0 + \mathcal{K}_k\f$.

### A-Conjugacy

Two vectors are **A-conjugate** if \f$\mathbf{p}^T A\mathbf{q} = 0\f$.  Minimizing
\f$\phi\f$ along an A-conjugate direction leaves progress along previous
directions undisturbed.  CG maintains A-conjugacy using only the immediately
preceding direction -- a consequence of the three-term Lanczos recurrence.

### Step Length and Direction Update

At iteration \f$k\f$, exact line minimization along the search direction
\f$\mathbf{p}_k\f$ gives

\f[
\alpha_k = \frac{\mathbf{r}_k^T\mathbf{r}_k}{\mathbf{p}_k^T A \mathbf{p}_k}
\f]

After updating \f$\mathbf{x}_{k+1} = \mathbf{x}_k + \alpha_k\mathbf{p}_k\f$ and
\f$\mathbf{r}_{k+1} = \mathbf{r}_k - \alpha_k A\mathbf{p}_k\f$, the new A-conjugate
direction is

\f[
\mathbf{p}_{k+1} = \mathbf{r}_{k+1} + \beta_k\mathbf{p}_k, \qquad
\beta_k = \frac{\mathbf{r}_{k+1}^T\mathbf{r}_{k+1}}{\mathbf{r}_k^T\mathbf{r}_k}
\f]

The dominant cost per iteration is the matrix-vector product
\f$A\mathbf{p}_k\f$ -- \f$O(n^2)\f$ for dense \f$A\f$, \f$O(\mathrm{nnz})\f$ for sparse.

### Convergence

CG terminates in at most \f$n\f$ iterations in exact arithmetic.  In practice:

\f[
\frac{\|\mathbf{e}_k\|_A}{\|\mathbf{e}_0\|_A} \leq 2\left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^k
\f]

where \f$\kappa = \kappa_2(A) = \lambda_{\max}/\lambda_{\min}\f$.

| \f$\kappa\f$ | Iterations for 6-digit accuracy |
|------------|--------------------------------|
| \f$10\f$ | ~7 |
| \f$100\f$ | ~23 |
| \f$10^4\f$ | ~230 |
| \f$10^6\f$ | ~2300 |

Eigenvalue **clustering** accelerates convergence: if all eigenvalues except
a small cluster lie in a single interval, CG quickly annihilates the
corresponding polynomial and then converges as if the problem had only as
many distinct eigenvalues as the cluster size.

### Preconditioned CG (PCG)

Replace \f$A\f$ with \f$M^{-1}A\f$ where \f$M \approx A\f$ is cheap to invert.
The PCG scalars change to

\f[
\alpha_k = \frac{\mathbf{r}_k^T\mathbf{z}_k}{\mathbf{p}_k^T A\mathbf{p}_k}, \qquad
\beta_k  = \frac{\mathbf{r}_{k+1}^T\mathbf{z}_{k+1}}{\mathbf{r}_k^T\mathbf{z}_k}
\f]

where \f$\mathbf{z}_k = M^{-1}\mathbf{r}_k\f$ is the preconditioner solve.
Common preconditioners: diagonal (Jacobi), incomplete Cholesky (IC0), and
algebraic multigrid (AMG, giving \f$\kappa(M^{-1}A) \approx O(1)\f$ for
Poisson-type problems).

### Matrix-Free CG

`cg_matfree` accepts a \f$\mathtt{MatVecFn}\f$ callback instead of an explicit
matrix, enabling implicit time stepping (\f$A = M + \Delta t\, K\f$, never
assembled), spectral methods (apply \f$A\f$ via FFT in \f$O(n\log n)\f$), and
structured operators (Toeplitz, circulant) with \f$O(n)\f$ storage.

### API

- @ref num::cg -- conjugate gradient for SPD systems (dense or sparse)
- @ref num::cg_matfree -- matrix-free CG with a callback matvec

---

## GMRES {#sec_gmres}

### Problem and Approach

GMRES solves \f$Ax = b\f$ for any invertible \f$A\f$, including non-symmetric
matrices where CG does not apply.  At step \f$k\f$ it finds

\f[
x_k = \operatorname*{argmin}_{x \in x_0 + \mathcal{K}_k}\|Ax - b\|_2
\f]

over the same Krylov subspace \f$\mathcal{K}_k(A,r_0)\f$.

### Arnoldi Relation

The Krylov basis is built by the Arnoldi process using Modified Gram-Schmidt
(MGS).  After \f$k\f$ steps:

\f[
A V_k = V_{k+1}\bar{H}_k
\f]

where \f$V_k = [v_1 \mid \cdots \mid v_k]\f$ has orthonormal columns and
\f$\bar{H}_k \in \mathbb{R}^{(k+1)\times k}\f$ is the upper Hessenberg matrix
of MGS coefficients.  The MGS step at iteration \f$j\f$ is:

\f[
w \leftarrow Av_j, \qquad
h_{ij} = w^T v_i, \quad w \leftarrow w - h_{ij} v_i \quad (i = 0,\ldots,j),
\qquad
v_{j+1} = w / \|w\|
\f]

### The GMRES Least-Squares Problem

Using the Arnoldi relation and orthonormality of \f$V_{k+1}\f$:

\f[
\|Ax_k - b\|_2 = \|\bar{H}_k y_k - \beta e_1\|_2, \qquad \beta = \|r_0\|_2
\f]

This is a small \f$(k+1) \times k\f$ least-squares problem solved
**incrementally** via Givens rotations.  At step \f$j\f$ a new rotation
annihilates the sub-diagonal entry \f$H_{j,j+1}\f$:

\f[
\begin{pmatrix} c_j & s_j \\ -s_j & c_j \end{pmatrix}
\begin{pmatrix} H_{jj} \\ H_{j,j+1} \end{pmatrix}
= \begin{pmatrix} d \\ 0 \end{pmatrix},
\qquad
c_j = \frac{H_{jj}}{d},\quad s_j = \frac{H_{j,j+1}}{d},\quad
d = \sqrt{H_{jj}^2 + H_{j,j+1}^2}
\f]

The rotated right-hand side vector \f$g\f$ satisfies \f$|g_{j+1}| = \|r_{j+1}\|_2\f$,
giving the residual norm **without computing \f$x_k\f$ explicitly** -- used as the
convergence monitor.

Once \f$\|r_k\|_2 < \mathrm{tol}\f$, back-solve the \f$k \times k\f$ upper
triangular system \f$H_{[k]}y = g_{[k]}\f$ and recover
\f$x_k = x_0 + V_k y\f$.

### Restart: GMRES(\f$m\f$)

Full GMRES stores \f$k\f$ vectors growing without bound.
**Restarted GMRES(\f$m\f$)** caps memory at \f$m+1\f$ vectors of length \f$n\f$:
after \f$m\f$ Arnoldi steps, update \f$x_0\f$ from the current solution and
restart.

- **Memory**: \f$O(mn)\f$. Typical restart values: \f$m \in [30, 300]\f$.
- **FLOPs per restart cycle**: \f$2mn^2\f$ (matvecs) \f$+ 2m^2n\f$ (MGS)
  \f$+ O(m^2)\f$ (Givens). For \f$m \ll n\f$ the matvec dominates.
- **When to prefer GMRES over CG**: whenever \f$A\f$ is non-symmetric or
  indefinite. For SPD systems CG is cheaper (short recurrence, \f$O(n)\f$
  memory per iteration vs \f$O(mn)\f$).

### Convergence

GMRES convergence is governed by the polynomial approximation problem:

\f[
\frac{\|r_k\|_2}{\|r_0\|_2} \leq
\min_{\substack{p_k \in \mathcal{P}_k \\ p_k(0)=1}}
\max_{\lambda \in \sigma(A)} |p_k(\lambda)|
\f]

Convergence is fast when the spectrum \f$\sigma(A)\f$ is clustered away from
the origin; it can stagnate for many steps then converge suddenly if one
eigenvalue is an outlier far from the main cluster.

Preconditioning replaces \f$A\f$ by \f$M^{-1}A\f$ (left) or \f$AM^{-1}\f$ (right);
right preconditioning is usually preferred because it minimizes the true
residual norm \f$\|r_k\|_2\f$ rather than the preconditioned norm
\f$\|M^{-1}r_k\|_2\f$.

### API

- @ref num::gmres -- restarted GMRES(\f$m\f$) for general non-singular systems
