# Dense Matrix Factorizations {#page_factorizations}

A matrix factorization rewrites \f$A\f$ as a product of simpler matrices whose structure reduces solving, least-squares, and determinant problems to cheap triangular operations.

---

## Why Factorizations? {#sec_factorization_motivation}

Computing \f$A^{-1}\f$ and then \f$x = A^{-1}b\f$ costs two \f$O(n^3)\f$ operations.
Factorization costs one \f$O(n^3)\f$ pass up front; each subsequent solve
(forward + backward substitution) costs only \f$O(n^2)\f$.

When the same coefficient matrix appears with many right-hand sides--time
steps of an ODE, Newton iterations, column-by-column inversion--the
factorization is amortised:

\f[
\underbrace{O(n^3)}_{\text{factor once}} + k \cdot \underbrace{O(n^2)}_{\text{solve}}
\quad \ll \quad k \cdot O(n^3)
\f]

For \f$k = n\f$ solves the factorisation approach is \f$n\f$ times cheaper.

---

## LU Factorization with Partial Pivoting {#sec_lu}

### The PA = LU Form

Every non-singular \f$A \in \mathbb{R}^{n \times n}\f$ admits

\f[
PA = LU
\f]

where:
- **P** is a permutation matrix (records row swaps),
- **L** is unit lower triangular (\f$L_{ii} = 1\f$, \f$|L_{ij}| \leq 1\f$ with pivoting),
- **U** is upper triangular.

\f$L\f$ and \f$U\f$ are stored packed in a single \f$n \times n\f$ working matrix:
\f$U\f$ occupies the diagonal and above; \f$L\f$ occupies the strict lower
triangle with the implicit unit diagonal.

### Doolittle Algorithm

Gaussian elimination proceeds column by column. At step \f$k\f$, the multipliers

\f[
l_{ik} = \frac{a_{ik}^{(k)}}{a_{kk}^{(k)}}, \quad i = k+1, \ldots, n-1
\f]

are stored in the lower triangle, and the trailing submatrix receives a
rank-1 (Schur complement) update:

\f[
a_{ij}^{(k+1)} = a_{ij}^{(k)} - l_{ik}\, a_{kj}^{(k)}, \quad i,j > k
\f]

After \f$n-1\f$ steps the working matrix holds \f$L\f$ (below diagonal) and
\f$U\f$ (diagonal and above).

### Why Partial Pivoting?

Without pivoting, a tiny pivot \f$a_{kk}^{(k)}\f$ inflates the multipliers
catastrophically. Example: for

\f[
A = \begin{pmatrix} \varepsilon & 1 \\ 1 & 1 \end{pmatrix}, \quad \varepsilon = 10^{-16}
\f]

without pivoting \f$l_{21} \approx 10^{16}\f$ and \f$U_{22} \approx -10^{16}\f$.
Swapping rows first gives \f$l_{21} = \varepsilon\f$ and \f$U_{22} \approx 1\f$.

**Partial pivoting** -- swapping row \f$k\f$ with the row achieving
\f$\max_{i \geq k}|a_{ik}^{(k)}|\f$ -- guarantees \f$|L_{ij}| \leq 1\f$ for all
\f$i > j\f$, which bounds the **growth factor**:

\f[
\|LU\|_\infty \leq 2^{n-1} \|A\|_\infty
\f]

The \f$2^{n-1}\f$ bound is the Wilkinson worst case; random matrices grow as
\f$O(n^{1/2})\f$ in practice, and diagonally dominant or SPD matrices have a
growth factor bounded by a small constant.

### Solving with the Factorization

Given \f$PA = LU\f$ and a right-hand side \f$b\f$, solve \f$Ax = b\f$ in three steps:

1. **Apply permutation**: \f$y \leftarrow Pb\f$ (row swaps, no arithmetic).
2. **Forward substitution** \f$Ly = Pb\f$ (unit diagonal, \f$n^2\f$ flops):
\f[
y_i = y_i - \sum_{j=0}^{i-1} L_{ij}\, y_j, \quad i = 1, \ldots, n-1
\f]
3. **Backward substitution** \f$Ux = y\f$ (\f$n^2\f$ flops):
\f[
x_i = \frac{1}{U_{ii}}\!\left(y_i - \sum_{j=i+1}^{n-1} U_{ij}\, x_j\right), \quad i = n-1, \ldots, 0
\f]

Each triangular solve costs \f$n^2\f$ flops; the factorization itself costs
\f$\frac{2}{3}n^3\f$ flops.

### Determinant and Inverse

\f[
\det(A) = (-1)^{\text{swaps}} \prod_{i=0}^{n-1} U_{ii}
\f]

where "swaps" counts non-trivial pivots (\f$\text{piv}[k] \neq k\f$).

The inverse is computed by solving \f$AX = I\f$ column by column--\f$n\f$ calls
to the triangular solve, each \f$O(n^2)\f$, total \f$O(n^3)\f$.  Never compute
\f$A^{-1}\f$ explicitly when only \f$A^{-1}b\f$ is needed; solve \f$Ax = b\f$
directly.

### Complexity and Stability

| Operation | FLOPs | Notes |
|-----------|-------|-------|
| LU factorization | \f$\frac{2}{3}n^3\f$ | Dominant one-time cost |
| Forward substitution | \f$n^2\f$ | Per right-hand side |
| Backward substitution | \f$n^2\f$ | Per right-hand side |

LU with partial pivoting is backward stable: the computed solution
\f$\hat{x}\f$ satisfies

\f[
(A + \delta A)\,\hat{x} = b, \qquad
\frac{\|\delta A\|}{\|A\|} \leq \varepsilon_{\mathrm{mach}}\,\rho(n)\,\kappa(A)
\f]

where \f$\rho(n) \leq 2^{n-1}\f$ is the growth factor and
\f$\kappa(A) = \|A\|\,\|A^{-1}\|\f$ is the condition number.

### API

- @ref num::lu -- compute the \f$PA = LU\f$ factorization
- @ref num::lu_solve -- forward/backward substitution given a factored system
- @ref num::lu_det -- determinant from the packed factorization
- @ref num::lu_inv -- matrix inverse via \f$n\f$ triangular solves

---

## QR Factorization (Householder) {#sec_qr}

### The A = QR Form

Every \f$A \in \mathbb{R}^{m \times n}\f$ with \f$m \geq n\f$ admits

\f[
A = QR
\f]

where \f$Q \in \mathbb{R}^{m \times m}\f$ is orthogonal (\f$Q^TQ = I\f$) and
\f$R \in \mathbb{R}^{m \times n}\f$ is upper triangular.  The **economy (thin)**
form uses \f$\hat{Q} \in \mathbb{R}^{m \times n}\f$ and
\f$\hat{R} \in \mathbb{R}^{n \times n}\f$, discarding the \f$m - n\f$ trailing
columns of \f$Q\f$ that contribute only to the residual.

### Householder Reflectors

A Householder reflector is the rank-1 symmetric orthogonal matrix

\f[
H = I - 2\hat{v}\hat{v}^T, \qquad H^T = H, \quad H^2 = I, \quad \det(H) = -1
\f]

For a vector \f$x \in \mathbb{R}^m\f$, choose \f$\hat{v}\f$ so that \f$Hx\f$ is a
scalar multiple of \f$e_1\f$:

\f[
Hx = -\operatorname{sign}(x_0)\|x\|_2\, e_1
\f]

**Construction**: form the unnormalized vector

\f[
v = x + \operatorname{sign}(x_0)\|x\|_2\, e_1, \qquad \hat{v} = v / \|v\|
\f]

**Sign convention**: choose the sign *matching* \f$x_0\f$, so that
\f$v_0 = x_0 + \operatorname{sign}(x_0)\|x\|\f$ is a sum of same-sign terms.
The alternative \f$v_0 = x_0 - \|x\|\f$ suffers catastrophic cancellation when
\f$x_0 \approx \|x\|\f$.  Concisely:

\f[
v_0 \mathrel{+}= \operatorname{sign}(x_0)\|x\|
\f]

### Algorithm

Apply \f$r = \min(m-1, n)\f$ reflections. Step \f$k\f$ zeros all entries of
column \f$k\f$ strictly below the diagonal by applying
\f$H_k = I - 2\hat{v}_k\hat{v}_k^T\f$ to the trailing submatrix
\f$R[k:m,\, k:n]\f$:

\f[
R[k:m,\, j] \;\leftarrow\; R[k:m,\, j] - 2\hat{v}_k\!\left(\hat{v}_k^T R[k:m,\, j]\right),
\quad j = k, \ldots, n-1
\f]

After \f$r\f$ steps, \f$R\f$ is upper triangular.  The full \f$Q\f$ is recovered by
accumulating reflectors in **reverse** order starting from \f$Q = I_m\f$:

\f[
Q \;\leftarrow\; H_0 H_1 \cdots H_{r-1}
\f]

(built as \f$Q \leftarrow H_{r-1}, \ldots, H_0\f$ applied right-to-left).

If only \f$Q^Tb\f$ is required for a least-squares solve, apply each \f$H_k\f$
directly to \f$b\f$ in \f$O(mn)\f$ total -- never form \f$Q\f$ explicitly.

### Why Householder Beats Gram-Schmidt

Classical Gram-Schmidt orthogonalizes columns sequentially and is not
backward stable: round-off in early columns corrupts later ones, and for
ill-conditioned matrices the computed \f$Q\f$ can be far from orthogonal.

Householder is **backward stable** regardless of \f$\kappa(A)\f$:

\f[
(A + \delta A) = \hat{Q}\hat{R}, \qquad
\frac{\|\delta A\|_F}{\|A\|_F} = O(\varepsilon_{\mathrm{mach}})
\f]

The normal equations \f$A^TAx = A^Tb\f$ satisfy
\f$\|\delta A\|/\|A\| = O(\varepsilon_{\mathrm{mach}}\,\kappa(A)^2)\f$;
for \f$\kappa(A) > 10^8\f$ in double precision they lose all significant
digits while QR still gives correct residuals.

### Least-Squares Solve

For the overdetermined system \f$\min_x \|Ax - b\|_2\f$ with \f$m > n\f$, use
the isometry of \f$Q\f$:

\f[
\|Ax - b\|_2^2 = \|QRx - b\|_2^2 = \|Rx - Q^Tb\|_2^2
= \|\hat{R}x - \tilde{c}\|_2^2 + \|\bar{c}\|_2^2
\f]

where \f$c = Q^Tb\f$, \f$\tilde{c} = c[0:n]\f$ (the matched component), and
\f$\bar{c} = c[n:m]\f$ (the unavoidable residual).  The minimum is attained by
solving the \f$n \times n\f$ upper triangular system

\f[
\hat{R}\,x = \tilde{c}
\f]

via backward substitution.  The residual norm \f$\|\bar{c}\|_2\f$ is available
as a by-product at no extra cost.

### Complexity

| Operation | FLOPs | Notes |
|-----------|-------|-------|
| Factorization | \f$2mn^2 - \frac{2}{3}n^3\f$ | Building \f$R\f$ and storing reflectors |
| Building \f$Q\f$ explicitly | \f$4m^2n - 2mn^2 + \frac{2}{3}n^3\f$ | Avoid when possible |
| Applying \f$Q^T\f$ to a vector | \f$2mn\f$ | Sufficient for least-squares |
| Backward substitution | \f$n^2\f$ | Per right-hand side |

### API

- @ref num::qr -- compute the Householder QR factorization
- @ref num::qr_solve -- least-squares solve \f$\min\|Ax - b\|_2\f$ given a factored system
