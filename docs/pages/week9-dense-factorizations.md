# Week 9: Dense Matrix Factorizations -- LU and QR {#page_week9}

## Overview

A **matrix factorization** decomposes A into a product of simpler matrices
whose structure makes solving linear systems, least-squares problems, and
determinants straightforward.  This week we implement two fundamental ones:

| Factorization | Form | What it solves |
|---|---|---|
| **LU** (Gaussian elimination) | PA = LU | Square systems Ax = b |
| **QR** (Householder) | A = QR | Least-squares min ||Ax-b|| |

Both are O(n^3) in FLOPs and are the workhorses for everything built later:
eigenvalue solvers, Newton's method, ODE implicit steps, and FEM assembly.

---

## 1. Why Factorizations?

Solving Ax = b by computing A^-^1 and then x = A^-^1b costs **two** O(n^3)
operations.  Factorization costs **one** O(n^3) pass; each subsequent solve
(forward + backward substitution) costs only O(n^2).

When you need to solve Ax = bi for many right-hand sides b_1, b_2, ... (e.g. at
each time step of an ODE, or for computing the inverse column by column), the
factorization is done once and amortised over all solves.

```
Naive (A^-^1 per solve):    k solves -> k x O(n^3) = O(k n^3)
Factorization then solve:             O(n^3) + k x O(n^2)
```

For k = n, factorization is nx cheaper.

---

## 2. LU Factorization with Partial Pivoting

### The idea: Gaussian elimination with bookkeeping

Gaussian elimination zeros below-diagonal entries column by column.  At step
k it subtracts multiples of row k from rows below it.  The multipliers
l_{ik} = A(i,k)/A(k,k) are exactly the entries of the lower triangular factor
L.  After n-1 steps the matrix is upper triangular U.

\f[PA = LU\f]

where:
- **P** is a permutation matrix (row swaps)
- **L** is unit lower triangular (diagonal = 1, |L(i,j)| <= 1 with pivoting)
- **U** is upper triangular

### Why partial pivoting?

Without pivoting, if A(k,k) = 0 the algorithm divides by zero.  If A(k,k) is
small, the multipliers l_{ik} blow up and amplify round-off errors.

**Partial pivoting**: at step k, swap row k with the row that has the largest
|A(i,k)| for i >= k.  This guarantees |L(i,j)| <= 1 for all i > j, bounding
the **growth factor** -- the ratio of largest element after elimination to
before -- to at most 2^{n-1}.  In practice, growth is typically O(n^{2/3}).

### Storage: packed L and U

L and U are stored in a single nxn matrix: U occupies the diagonal and above,
L occupies the strict lower triangle (the diagonal of L is 1 implicitly).

```
A = [2  1  0]        After LU:   LU = [4   3   2 ]   piv = [1, 2, 2]
    [4  3  2]                         [0.5 0.5 -1]
    [8  7  9]                         [2   1   3 ]
```

### Solving with LU (`src/factorization/lu.cpp:54-83`)

Given PA = LU and the system Ax = b:

1. **Apply permutation**: y = Pb (swap rows of b using piv)
2. **Forward substitution**: solve Lz = y (L has 1s on diagonal -- no division)
3. **Backward substitution**: solve Ux = z

```cpp
// Forward substitution: L z = y (L's diagonal is 1, implicit)
for (idx i = 1; i < n; ++i)
    for (idx j = 0; j < i; ++j)
        y[i] -= M(i, j) * y[j];

// Backward substitution: U x = z
for (idx i = n; i-- > 0; ) {
    for (idx j = i + 1; j < n; ++j)
        y[i] -= M(i, j) * y[j];
    y[i] /= M(i, i);
}
```

Each step is O(n^2); the factorisation itself is O(n^3).

### Determinant and inverse

```
det(A) = det(P)^-^1 * det(L) * det(U) = (-1)^{swaps} * 1 * prod U[i,i]
```

The inverse is obtained by solving AX = I column by column -- n solves of cost
O(n^2) each, total O(n^3).  Only compute the inverse when you genuinely need it;
for specific right-hand sides, `lu_solve` is cheaper.

---

## 3. QR Factorization via Householder Reflections

### The idea

QR factorises A as A = QR where Q is orthogonal (Q^T Q = I) and R is upper
triangular.  For the least-squares problem min||Ax - b||:

\f[\|Ax - b\|^2 = \|QRx - b\|^2 = \|Rx - Q^Tb\|^2\f]

since Q is isometric (preserves norms).  This splits into a triangular solve
plus a fixed residual.

### Why Householder instead of Gram-Schmidt?

Classical Gram-Schmidt (CGS) orthogonalises columns sequentially.  It is
**forward-stable** but not **backward-stable**: round-off errors in early
columns corrupt later ones.  For ill-conditioned matrices CGS can produce
nearly non-orthogonal Q.

A **Householder reflector** H = I - 2vv^T is orthogonal by construction and
acts on the entire working matrix at once.  Householder QR is backward-stable:
the computed factorisation satisfies (A + E) = Q R with ||E||/||A|| = O(u),
where u is machine epsilon.

### The Householder reflector

For a vector x, choose v so that Hx = -sign(x_0)||x|| e_1:

\f[v = x + \text{sign}(x_0)\|x\| e_1, \qquad \hat{v} = v/\|v\|\f]

The sign choice is critical: if x_0 > 0 and we used v_0 = x_0 - ||x||, then
v_0 -> 0 as x_0 -> ||x|| (catastrophic cancellation).  Taking the **same sign**
as x_0 avoids this.

```cpp
// Sign trick: add, don't subtract
v[0] += (x[0] >= real(0)) ? norm_x : -norm_x;
```

### Algorithm (`src/factorization/qr.cpp:65-103`)

```
for k = 0 .. min(m-1, n) - 1:
    x  <- R[k:m, k]                        (column k from row k down)
    v  <- x + sign(x[0]) * ||x|| * e_1    (Householder vector)
    v  <- v / ||v||                         (normalise)
    for j = k .. n-1:                      (apply H_k to each column)
        R[k:m, j] -= 2*v*(v^T * R[k:m, j])
    save v for Q accumulation
```

After r = min(m-1, n) steps, R is upper triangular.

### Building Q explicitly

Q = H_0 * H_1 * ... * H_{r-1}.  Build by accumulating in reverse:

```
Q = I
for k = r-1 downto 0:
    Q[k:m, k:m] = H_k * Q[k:m, k:m]
```

**Why reverse?**  At step k (going backwards), Q[:,j] = ej for j < k -- those
columns have not been touched yet.  So the H_k update only needs j >= k.

### Least-squares solve (`src/factorization/qr.cpp:119-136`)

```cpp
// y = Q^T b  (Q^T[i,j] = Q[j,i])
for (idx i = 0; i < m; ++i)
    for (idx j = 0; j < m; ++j)
        y[i] += f.Q(j, i) * b[j];

// Back-substitute: R[:n,:n] x = y[:n]
for (idx i = n; i-- > 0; ) {
    xv[i] = y[i];
    for (idx j = i + 1; j < n; ++j) xv[i] -= f.R(i, j) * xv[j];
    xv[i] /= f.R(i, i);
}
```

The residual ||(Q^T b)[n:]|| is available for free -- no need for an extra
matrix-vector product.

### Sign of R's diagonal

Householder QR does not guarantee positive diagonal in R.  Each reflector has
determinant -1, so some diagonal entries of R may be negative.  This is
mathematically correct -- both Q and R differ from the "canonical" form by
sign flips.  Q*R = A holds exactly regardless.

If positive-diagonal R is required (e.g., for uniqueness of the factorisation
when A has full column rank), flip the sign of each negative R[k,k] and the
corresponding column of Q.  We omit this normalisation here.

---

## 4. API Reference

### Declarations (`include/factorization/`)

```cpp
// -- LU -----------------------------------------------------------------------
struct LUResult {
    Matrix           LU;        // packed L (below diag) + U (diag + above)
    std::vector<idx> piv;       // piv[k] = row swapped to position k
    bool             singular;
};

LUResult lu(const Matrix& A);
void     lu_solve(const LUResult&, const Vector& b, Vector& x);
void     lu_solve(const LUResult&, const Matrix& B, Matrix& X);  // multi-RHS
real     lu_det  (const LUResult&);
Matrix   lu_inv  (const LUResult&);

// -- QR -----------------------------------------------------------------------
struct QRResult {
    Matrix Q;   // mxm orthogonal
    Matrix R;   // mxn upper triangular
};

QRResult qr      (const Matrix& A);
void     qr_solve(const QRResult&, const Vector& b, Vector& x);
```

### Usage example

```cpp
#include "numerics.hpp"
using namespace num;

// LU: solve Ax = b
Matrix A = ...;
Vector b = ...;
auto  f  = lu(A);
Vector x(n);
lu_solve(f, b, x);

// Solve again with a different b (only O(n^2) this time)
Vector x2(n);
lu_solve(f, b2, x2);

// Determinant
real d = lu_det(f);

// QR: least-squares fit
Matrix A_ls = ...;   // overdetermined mxn
Vector b_ls = ...;
auto   fq   = qr(A_ls);
Vector x_ls(n);
qr_solve(fq, b_ls, x_ls);
```

---

## 5. What These Unlock

| Future module | Uses |
|---|---|
| Eigenvalue solvers | QR iteration (apply QR repeatedly to converge to Schur form) |
| Newton's method (optimisation) | LU solve at each step: `H * Deltax = -g` |
| ODE implicit solvers | LU for local Jacobian solve at each time step |
| FEM | LU/CG for assembled stiffness matrix |
| Linear regression | QR least-squares |
| Pseudo-inverse / SVD | QR as intermediate step |

---

## 6. Complexity and Numerical Properties

| Operation | FLOPs | Stability |
|---|---|---|
| LU factorisation | 2n^3/3 | Backward-stable with partial pivoting |
| Forward sub (Ly=b) | n^2 | Exact (no cancellation with ||L||<=1) |
| Backward sub (Ux=y) | n^2 | Backward-stable |
| QR factorisation | 2mn^2 - 2n^3/3 (m>=n) | Backward-stable |
| Q^T b multiply | 2mn | Backward-stable |
| Backward sub for QR | n^2 | Backward-stable |

LU growth factor worst case: 2^{n-1}.  In practice for random matrices: O(n^{1/2}).
For diagonally dominant or SPD matrices, growth is bounded by a small constant.

---

## 7. Key Takeaways

1. **Factorise once, solve many times.**  The O(n^3) cost is paid upfront; each
   subsequent right-hand side costs only O(n^2).

2. **Partial pivoting makes LU stable.**  Without it, tiny pivots amplify
   round-off to catastrophic levels.  |L(i,j)| <= 1 bounds the growth factor.

3. **Householder is more stable than Gram-Schmidt.**  It is backward-stable
   regardless of condition number; CGS loses orthogonality for ill-conditioned
   matrices.

4. **QR naturally gives least squares.**  No need to form the normal equations
   A^T A x = A^T b (which squares the condition number); QR solves in the
   original norm.

5. **The sign of R's diagonal is not uniquely determined.**  Q*R = A always
   holds; normalisation to positive diagonal is optional.

---

## Exercises

1. Solve a 4x4 system by hand using Gaussian elimination with partial pivoting.
   Verify that your L satisfies |L(i,j)| <= 1 everywhere.

2. Compare `lu_solve` and `cg` on a 100x100 SPD system.  Which is faster?
   When would you prefer CG over LU for SPD systems?

3. Use `lu_inv` to compute A^-^1 for a 3x3 matrix and verify A * A^-^1 = I.
   Why is computing the inverse wasteful if you only need one solve?

4. Implement `lu_log_det`: compute log|det A| without overflow by summing
   log|U[i,i]| instead of multiplying.  When is this needed?

5. *(Advanced)* Implement Modified Gram-Schmidt (MGS) QR and compare its
   orthogonality error ||Q^T Q - I|| against Householder QR for an
   ill-conditioned Vandermonde matrix.

---

## References

- Trefethen, L. & Bau, D. (1997). *Numerical Linear Algebra.* SIAM.
  -- Lectures 20-23 (Householder), 21 (stability analysis). The standard reference.
- Golub, G. & Van Loan, C. (2013). *Matrix Computations*, 4th ed. Johns Hopkins.
  -- Sec.3.2 (LU), Sec.5.1 (Householder). Comprehensive.
- Higham, N. (2002). *Accuracy and Stability of Numerical Algorithms*, 2nd ed. SIAM.
  -- Sec.9 (Gaussian elimination stability), Sec.19 (orthogonal factorizations).
