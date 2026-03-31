# LU Factorization with Partial Pivoting {#page_lu_notes}

## Overview

Factors a non-singular \f$A \in \mathbb{R}^{n \times n}\f$ as

\f[PA = LU\f]

where \f$P\f$ is a permutation, \f$L\f$ is unit lower triangular, and \f$U\f$ is upper triangular. **Partial pivoting** selects the pivot row to maximize \f$|U_{kk}|\f$ at each step, keeping \f$|L_{ij}| \leq 1\f$ for all \f$i > j\f$ and preventing catastrophic cancellation.

---

## Algorithm

**Doolittle LU with partial pivoting** stores \f$L\f$ (below diagonal) and \f$U\f$ (diagonal and above) in the same working matrix \f$M\f$. The diagonal of \f$L\f$ is implicitly 1.

```
function LU(A):
    n <- A.rows
    M <- copy of A
    piv[0..n-1] <- identity permutation

    for k = 0 to n-1:
        // Partial pivot: find row with max |M[i,k]| for i >= k
        pivot_row <- argmax_{i >= k} |M[i,k]|
        piv[k] <- pivot_row
        swap rows k and pivot_row of M

        if |M[k,k]| < eps: mark singular; continue

        // Column k of L: multipliers stored below diagonal
        for i = k+1 to n-1:
            M[i,k] <- M[i,k] / M[k,k]

        // Schur complement (rank-1 update of trailing submatrix)
        for i = k+1 to n-1:
            lik <- M[i,k]
            for j = k+1 to n-1:
                M[i,j] <- M[i,j] - lik * M[k,j]

    return (M, piv)
```

**Complexity**: \f$\frac{2}{3}n^3\f$ flops for factorization.

---

## Forward/Backward Substitution

Given \f$PAx = b\f$, i.e., \f$LUx = Pb\f$:

**Step 1** -- apply permutation: \f$y \leftarrow Pb\f$.

**Step 2** -- forward substitution \f$Ly = Pb\f$ (unit diagonal):

\f[y_i = y_i - \sum_{j=0}^{i-1} L_{ij}\, y_j, \quad i = 1, \ldots, n-1\f]

**Step 3** -- backward substitution \f$Ux = y\f$:

\f[x_i = \frac{1}{U_{ii}}\!\left(y_i - \sum_{j=i+1}^{n-1} U_{ij}\, x_j\right), \quad i = n-1, \ldots, 0\f]

```
function LU_solve(M, piv, b):
    y <- copy of b
    for k = 0..n-1: swap y[k], y[piv[k]]   // permutation

    for i = 1..n-1:                          // forward
        for j = 0..i-1: y[i] -= M[i,j] * y[j]

    for i = n-1 downto 0:                    // backward
        for j = i+1..n-1: y[i] -= M[i,j] * y[j]
        y[i] /= M[i,i]

    return y
```

**Complexity**: \f$n^2\f$ flops for each sweep.

---

## Determinant and Inverse

\f[\det(A) = (-1)^{\text{swaps}} \prod_{i=0}^{n-1} U_{ii}\f]

where \f$\text{swaps}\f$ = number of non-trivial pivots (\f$\text{piv}[k] \neq k\f$).

**Inverse**: solve \f$AX = I\f$ column by column -- \f$n\f$ calls to `LU_solve`, each \f$O(n^2)\f$, total \f$O(n^3)\f$. Never compute \f$A^{-1}\f$ explicitly if only \f$A^{-1}b\f$ is needed; just solve \f$Ax = b\f$ directly.

---

## Stability

LU with partial pivoting is backward stable. The growth factor \f$\rho(n)\f$ is bounded at \f$2^{n-1}\f$ in theory but observed as \f$O(n^{1/2})\f$ in practice. Complete pivoting (swapping rows and columns) has tighter theoretical bounds but is rarely needed; partial pivoting suffices for virtually all practical matrices.

---

## Performance Optimization

### Current Implementation

Naive in-place Doolittle. The innermost Schur complement update

\f[M_{ij} \leftarrow M_{ij} - \ell_{ik}\, U_{kj}, \quad i,j > k\f]

is a rank-1 SAXPY over row \f$j\f$. With row-major storage, access to \f$M[k,j]\f$ (a row) is contiguous, but the multiplier column \f$M[\cdot,k]\f$ is strided -- one cache miss per row for large \f$n\f$.

### Level-3 BLAS: Blocked LU (LAPACK `dgetrf`)

Partition \f$A\f$ into panels of width \f$n_b\f$ columns (typically \f$n_b \in [64, 256]\f$):

\f[\begin{pmatrix} A_{11} & A_{12} \\ A_{21} & A_{22} \end{pmatrix} \xrightarrow{\text{factor}} \begin{pmatrix} L_{11} U_{11} & U_{12} \\ A_{21} & A_{22} - L_{21}U_{12} \end{pmatrix}\f]

```
for k = 0 to n step nb:
    panel_lu(A[k:n, k:k+nb])           // unblocked, O(n * nb^2)
    apply_pivots(A[k:n, k+nb:n])
    dtrsm(L[k:k+nb, k:k+nb], A[k:k+nb, k+nb:n])   // triangular solve
    dgemm(A[k+nb:n, k:k+nb],           // Schur complement: BLAS-3!
          A[k:k+nb, k+nb:n],
          A[k+nb:n, k+nb:n])
```

The Schur complement update is a `dgemm` with arithmetic intensity \f$O(n_b)\f$ -- reaches near-peak FLOP/s. The panel factorization is \f$O(n \cdot n_b^2)\f$ -- cheap relative to the trailing DGEMM.

**Block size selection**: \f$n_b\f$ should satisfy \f$2 n_b \times n_b \times 8\,\text{bytes} \lesssim L_2\f$ cache. For a 256 KB L2: \f$n_b \approx \sqrt{256\text{KB}/16} \approx 128\f$.

### SIMD for the Schur Complement

The rank-1 update inner loop is a DAXPY with FMA:

```cpp
for (idx j = k+1; j < n; j += 4) {
    __m256d vlik = _mm256_broadcast_sd(&lik);
    __m256d vukj = _mm256_loadu_pd(&M[k][j]);
    __m256d vmij = _mm256_loadu_pd(&M[i][j]);
    vmij = _mm256_fnmadd_pd(vlik, vukj, vmij);   // M[i,j] -= lik * M[k,j]
    _mm256_storeu_pd(&M[i][j], vmij);
}
```

FMA throughput on AVX2: \f$2 \times 4 = 8\f$ flops/cycle. Align rows to 32-byte boundaries and use `_mm256_load_pd` instead of `_mm256_loadu_pd` for maximum throughput.

### Recursive LU

Use divide-and-conquer at split \f$n/2\f$ rather than a fixed panel width:

```
LU_recursive(A[0:n, 0:n]):
    LU_recursive(A[0:n/2, 0:n/2])              // left half
    dtrsm + dgemm for right block
    LU_recursive(A[n/2:n, n/2:n])              // Schur complement
```

This achieves optimal cache reuse at all levels of the hierarchy without needing to tune \f$n_b\f$ -- recursion bottoms out when the block fits in L1. Used in PLASMA and other communication-avoiding libraries.

### Multiple Right-Hand Sides

For \f$m\f$ RHS, batch the triangular solves:

\f[L\,Y = PB, \quad U\,X = Y\f]

Each is a single `DTRSM(L, Y)` and `DTRSM(U, Y)` call -- triangular solve for a matrix, achieving BLAS-3 performance instead of \f$m\f$ separate `DTRSV` calls.

---

## Relation to Other Factorizations

| Factorization | Applies to | Cost | Stable |
|---|---|---|---|
| LU + partial pivoting | General square | \f$\frac{2}{3}n^3\f$ | Yes (growth factor bounded) |
| Cholesky (\f$LL^T\f$) | SPD | \f$\frac{1}{3}n^3\f$ | Yes (unconditionally) |
| \f$LDL^T\f$ | Symmetric indefinite | \f$\frac{1}{3}n^3\f$ | With diagonal pivoting |
| QR (Householder) | Rectangular \f$m \times n\f$ | \f$2mn^2 - \frac{2}{3}n^3\f$ | Backward stable |

For SPD systems (FEM/FDM stiffness matrices), always prefer **Cholesky**: half the flops, half the memory, guaranteed stability without pivoting.
