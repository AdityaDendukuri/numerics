# Week 5: Banded Matrix Solvers -- Theory and HPC Implementation {#page_week5}

## 1. Introduction

Banded matrices arise naturally in the discretization of differential equations, particularly in:

- **Radiative transfer**: Two-stream and multi-stream approximations (TUVX photolysis)
- **Finite differences**: Higher-order stencils produce wider bands
- **Finite elements**: Local basis functions create banded mass/stiffness matrices
- **Spline interpolation**: B-spline collocation matrices

A **banded matrix** \f$A \in \mathbb{R}^{n \times n}\f$ has bandwidth parameters \f$(k_l, k_u)\f$ where:
- \f$k_l\f$ = number of **lower** (sub-) diagonals
- \f$k_u\f$ = number of **upper** (super-) diagonals

The element \f$a_{ij} = 0\f$ whenever \f$j < i - k_l\f$ or \f$j > i + k_u\f$.

**Structure visualization** (pentadiagonal, \f$k_l = k_u = 2\f$):

\f[A = \begin{pmatrix}
\times & \times & \times & 0 & 0 & \cdots \\
\times & \times & \times & \times & 0 & \cdots \\
\times & \times & \times & \times & \times & \cdots \\
0 & \times & \times & \times & \times & \cdots \\
\vdots & & \ddots & \ddots & \ddots & \ddots
\end{pmatrix}\f]

**Key advantage**: Exploiting band structure reduces:
- Storage: \f$O(n \cdot (k_l + k_u + 1))\f$ instead of \f$O(n^2)\f$
- Solve complexity: \f$O(n \cdot k_l \cdot (k_l + k_u))\f$ instead of \f$O(n^3)\f$

---

## 2. Band Storage Formats

### 2.1 LAPACK Band Storage (Column-Major)

The standard format for band storage, used in LAPACK's `DGBTRF`/`DGBTRS`, stores diagonals in rows of a 2D array with leading dimension \f$\text{ldab} = 2k_l + k_u + 1\f$.

**Layout**: For matrix element \f$a_{ij}\f$ within the band:
\f[\text{AB}(k_l + k_u + i - j, \, j) = a_{ij}\f]

where AB is the band storage array with 0-based indexing.

**Visual representation** (for \f$n=5\f$, \f$k_l=2\f$, \f$k_u=1\f$):

```
Original matrix A:          Band storage AB (ldab=6, n=5):

| a00 a01  0   0   0  |     row 0: [ *   *   *   *   *  ]  (extra for fill-in)
| a10 a11 a12  0   0  |     row 1: [ *   *   *   *   *  ]  (extra for fill-in)
| a20 a21 a22 a23  0  |     row 2: [a01 a12 a23 a34  * ]  (upper diagonal)
|  0  a31 a32 a33 a34 |     row 3: [a00 a11 a22 a33 a44]  (main diagonal)
|  0   0  a42 a43 a44 |     row 4: [a10 a21 a32 a43  * ]  (lower diagonal 1)
                            row 5: [a20 a31 a42  *   * ]  (lower diagonal 2)
```

**Why extra \f$k_l\f$ rows?** During LU factorization with row pivoting, fill-in can occur in the upper triangle up to \f$k_l\f$ positions above the original upper bandwidth.

### 2.2 Memory Layout Benefits

Column-major storage (as in the layout above) ensures that:

1. **LU factorization** accesses memory sequentially when eliminating within a column
2. **Forward/back substitution** has good cache behavior
3. **Direct LAPACK compatibility** for fallback to optimized libraries

**Memory access pattern during elimination of column \f$j\f$**:

```cpp
// Eliminating column j affects rows j+1 to min(j+kl, n-1)
// These are stored contiguously in column j of AB
for (idx i = 1; i <= min(kl, n-j-1); ++i) {
    AB[kv + i + j*ldab] *= inv_pivot;  // Contiguous access
}
```

---

## 3. LU Factorization for Banded Matrices

### 3.1 Mathematical Foundation

For a banded matrix \f$A\f$ with bandwidths \f$(k_l, k_u)\f$, we compute \f$PA = LU\f$ where:
- \f$P\f$ is a permutation matrix (from partial pivoting)
- \f$L\f$ is unit lower triangular with at most \f$k_l\f$ sub-diagonals
- \f$U\f$ is upper triangular with at most \f$k_l + k_u\f$ super-diagonals (due to fill-in)

**Theorem** (Band Preservation): If \f$A\f$ has lower bandwidth \f$k_l\f$ and upper bandwidth \f$k_u\f$, then:
- \f$L\f$ has lower bandwidth \f$k_l\f$
- \f$U\f$ has upper bandwidth \f$k_l + k_u\f$ (worst case with pivoting)

**Proof**: During Gaussian elimination of column \f$j\f$:
- Without pivoting: row \f$i > j\f$ is modified by subtracting a multiple of row \f$j\f$
- The rightmost nonzero in row \f$j\f$ is at column \f$j + k_u\f$
- The rightmost nonzero in row \f$i\f$ (for \f$i \leq j + k_l\f$) is at column \f$i + k_u\f$
- After elimination, row \f$i\f$ can have nonzeros up to column \f$\max(j + k_u, i + k_u) = j + k_u\f$ (since \f$i > j\f$)

With pivoting, row \f$j\f$ may come from row \f$j + k_l\f$ at worst, creating fill-in up to column \f$(j + k_l) + k_u\f$ for original row \f$j\f$. But this is stored in the extra \f$k_l\f$ rows we allocated. \f$\square\f$

### 3.2 Algorithm

**Standard Gaussian elimination** adapted for band structure:

For column \f$j = 0, 1, \ldots, n-1\f$:

1. **Pivot selection** (partial pivoting):
   Search rows \f$j\f$ to \f$\min(j + k_l, n-1)\f$ for largest \f$|a_{ij}|\f$

2. **Row interchange**:
   If pivot row \f$p \neq j\f$, swap rows \f$p\f$ and \f$j\f$
   (Only columns \f$\max(0, j-k_u)\f$ to \f$\min(j+k_l, n-1)\f$ need swapping)

3. **Elimination**:
   For rows \f$i = j+1\f$ to \f$\min(j + k_l, n-1)\f$:
   - Compute multiplier: \f$l_{ij} = a_{ij} / a_{jj}\f$
   - Update row \f$i\f$: \f$a_{ik} \leftarrow a_{ik} - l_{ij} \cdot a_{jk}\f$ for \f$k = j+1\f$ to \f$\min(j + k_u, n-1)\f$

### 3.3 The Algorithm

```
Input: A in band storage (ldab x n), pivot array ipiv[n]
Output: LU factors in A, permutation in ipiv

kv = kl + ku  // Offset to main diagonal in band storage

for j = 0 to n-1:
    // 1. Find pivot in column j
    pivot_row = j
    max_val = |AB[kv, j]|

    for i = j+1 to min(j+kl, n-1):
        if |AB[kv + i - j, j]| > max_val:
            max_val = |AB[kv + i - j, j]|
            pivot_row = i

    ipiv[j] = pivot_row

    // 2. Check for singularity
    if AB[kv, j] == 0:
        return SINGULAR at row j

    // 3. Swap rows if needed
    if pivot_row != j:
        for col = max(0, j-ku) to min(j+kl, n-1):
            swap AB[kv + j - col, col] and AB[kv + pivot_row - col, col]

    // 4. Compute multipliers and eliminate
    pivot_val = AB[kv, j]
    num_elim = min(kl, n - j - 1)

    // Scale column (compute L multipliers)
    for i = 1 to num_elim:
        AB[kv + i, j] /= pivot_val

    // Update trailing submatrix
    for k = j+1 to min(j + ku, n-1):
        a_jk = AB[kv + j - k, k]
        if a_jk != 0:
            for i = 1 to num_elim:
                AB[kv + i + j - k, k] -= AB[kv + i, j] * a_jk
```

### 3.4 Solving After Factorization

Given \f$PA = LU\f$, solve \f$A\mathbf{x} = \mathbf{b}\f$ in three steps:

**Step 1: Apply permutation** (\f$\mathbf{b} \leftarrow P\mathbf{b}\f$)
```
for i = 0 to n-1:
    if ipiv[i] != i:
        swap b[i] and b[ipiv[i]]
```

**Step 2: Forward substitution** (solve \f$L\mathbf{y} = P\mathbf{b}\f$)
```
for j = 0 to n-1:
    if b[j] != 0:
        for i = j+1 to min(j + kl, n-1):
            b[i] -= AB[kv + i - j, j] * b[j]
```

**Step 3: Back substitution** (solve \f$U\mathbf{x} = \mathbf{y}\f$)
```
for j = n-1 down to 0:
    b[j] /= AB[kv, j]
    if b[j] != 0:
        for i = max(0, j - ku) to j-1:
            b[i] -= AB[kv + i - j, j] * b[j]
```

---

## 4. Complexity Analysis

### 4.1 Operation Counts

**LU Factorization**:

For column \f$j\f$, the number of operations is:
- Pivot search: \f$O(k_l)\f$ comparisons
- Row swap: \f$O(k_l + k_u)\f$ swaps (at most)
- Multiplier computation: \f$k_l\f$ divisions (for rows \f$j+1\f$ to \f$j+k_l\f$)
- Submatrix update: \f$k_l \times k_u\f$ multiply-adds

Summing over all columns:
\f[\text{FLOPs} = \sum_{j=0}^{n-1} O(k_l \cdot k_u) = O(n \cdot k_l \cdot k_u)\f]

More precisely, accounting for boundary effects:
\f[\text{FLOPs} \approx n \cdot k_l \cdot (k_l + k_u) + O(k_l^2 \cdot k_u)\f]

**Forward/Back Substitution**:

- Forward: \f$\sum_{j=0}^{n-1} O(k_l) = O(n \cdot k_l)\f$
- Back: \f$\sum_{j=0}^{n-1} O(k_u) = O(n \cdot k_u)\f$
- Total solve: \f$O(n \cdot (k_l + k_u))\f$

### 4.2 Comparison with Dense and Tridiagonal

| Method | Factor | Solve | Storage |
|--------|--------|-------|---------|
| Dense LU | \f$\frac{2}{3}n^3\f$ | \f$2n^2\f$ | \f$n^2\f$ |
| Banded LU | \f$n \cdot k_l \cdot (k_l + k_u)\f$ | \f$2n(k_l + k_u)\f$ | \f$(2k_l + k_u + 1) \cdot n\f$ |
| Tridiagonal (Thomas) | \f$8n\f$ | \f$5n\f$ | \f$4n\f$ |

**Example**: \f$n = 10000\f$, pentadiagonal (\f$k_l = k_u = 2\f$)
- Dense: \f$6.7 \times 10^{11}\f$ FLOPs for factor
- Banded: \f$10000 \times 2 \times 4 = 80000\f$ FLOPs for factor
- **Speedup**: \f$8.3 \times 10^6\f$ times faster!

### 4.3 Memory Bandwidth Analysis

For large \f$n\f$, the algorithm is typically **memory-bound** rather than compute-bound.

**Bytes per FLOP** (banded factorization):
- Each element accessed roughly once during elimination
- Storage: \f$(2k_l + k_u + 1) \cdot n \times 8\f$ bytes (double precision)
- FLOPs: \f$n \cdot k_l \cdot (k_l + k_u)\f$
- Ratio: \f$\frac{8(2k_l + k_u + 1)}{k_l(k_l + k_u)}\f$ bytes/FLOP

For pentadiagonal (\f$k_l = k_u = 2\f$): \f$\frac{8 \times 7}{2 \times 4} = 7\f$ bytes/FLOP

Modern CPUs achieve ~50 GFLOPS but only ~50 GB/s memory bandwidth. At 7 bytes/FLOP, we're limited to ~7 GFLOPS -- **memory bound**.

---

## 5. Numerical Stability

### 5.1 Partial Pivoting Guarantees

**Theorem**: LU factorization with partial pivoting satisfies:
\f[\|L\|_\infty \leq 1 + k_l\f]

for banded matrices, where the bound comes from at most \f$k_l\f$ nonzeros per column of \f$L\f$, each with magnitude \f$\leq 1\f$.

**Growth Factor**: The growth factor \f$\rho = \max_{ij}|u_{ij}| / \max_{ij}|a_{ij}|\f$ is bounded:
\f[\rho \leq 2^{k_l}\f]

This is much better than the \f$2^{n-1}\f$ worst-case for dense matrices.

### 5.2 Condition Number and Error

The computed solution \f$\hat{\mathbf{x}}\f$ satisfies:
\f[\frac{\|\hat{\mathbf{x}} - \mathbf{x}\|}{\|\mathbf{x}\|} \lesssim \kappa(A) \cdot \epsilon_{\text{mach}}\f]

where \f$\kappa(A) = \|A\| \cdot \|A^{-1}\|\f$ is the condition number.

**Condition number estimation** for banded matrices:
- Use 1-norm: \f$\|A\|_1 = \max_j \sum_i |a_{ij}|\f$ (maximum absolute column sum)
- Estimate \f$\|A^{-1}\|_1\f$ via Hager's algorithm (one forward/back solve)

### 5.3 Diagonal Dominance

**Definition**: \f$A\f$ is **diagonally dominant** if:
\f[|a_{ii}| \geq \sum_{j \neq i} |a_{ij}|, \quad \forall i\f]

with strict inequality for at least one row.

**Theorem**: For diagonally dominant banded matrices:
1. No pivoting is needed (diagonal is always the largest element in column)
2. No fill-in occurs beyond original bandwidth
3. The factorization is unconditionally stable

**Common diagonally dominant systems**:
- Laplacian discretization (FD, FEM)
- Diffusion equations
- Many radiative transfer discretizations

---

## 6. Implementation Details

### 6.1 BandedMatrix Class Design

```cpp
class BandedMatrix {
public:
    BandedMatrix(idx n, idx kl, idx ku);

    // Element access in original matrix coordinates
    real& operator()(idx i, idx j);

    // Direct band storage access
    real& band(idx band_row, idx col);

    // Properties
    idx size() const { return n_; }
    idx kl() const { return kl_; }
    idx ku() const { return ku_; }
    idx ldab() const { return 2*kl_ + ku_ + 1; }

private:
    idx n_, kl_, ku_, ldab_;
    std::unique_ptr<real[]> data_;  // Column-major: data_[band_row + col*ldab_]
};
```

**Element access mapping**:
```cpp
real& BandedMatrix::operator()(idx i, idx j) {
    // A(i,j) stored at band(kl + ku + i - j, j)
    return data_[(kl_ + ku_ + i - j) + j * ldab_];
}
```

### 6.2 SIMD Optimization

The inner loops in elimination and solve are vectorizable:

```cpp
// Multiplier scaling (forward elimination)
real inv_pivot = 1.0 / ab[kv + j * ldab];
#pragma omp simd
for (idx i = 1; i <= num_rows; ++i) {
    ab[kv + i + j * ldab] *= inv_pivot;
}

// Trailing submatrix update
#pragma omp simd
for (idx i = 1; i <= num_rows; ++i) {
    ab[kv + i - k + k * ldab] -= ab[kv + i + j * ldab] * a_jk;
}
```

The `#pragma omp simd` directive hints to the compiler that iterations are independent and can use SIMD instructions (AVX-512 on modern Intel/AMD).

### 6.3 Cache Optimization

**Blocking for L2 cache**: For very wide bands (\f$k_l, k_u > 100\f$), block the elimination:

```cpp
// Process columns in blocks of BLOCK_SIZE
for (idx jb = 0; jb < n; jb += BLOCK_SIZE) {
    idx je = min(jb + BLOCK_SIZE, n);

    // Factor diagonal block
    for (idx j = jb; j < je; ++j) {
        // ... standard elimination for column j ...
    }

    // Update remaining columns (can be parallelized)
    for (idx k = je; k < min(je + ku, n); ++k) {
        // Update column k using factored columns jb:je
    }
}
```

### 6.4 Multiple Right-Hand Sides

When solving with multiple RHS vectors (common in radiative transfer for multiple wavelengths):

```cpp
void banded_lu_solve_multi(const BandedMatrix& A, const idx* ipiv,
                           real* B, idx nrhs) {
    // B is n x nrhs, column-major

    #pragma omp parallel for if(nrhs > 16)
    for (idx rhs = 0; rhs < nrhs; ++rhs) {
        real* x = B + rhs * n;  // Column rhs

        // Apply permutation
        for (idx i = 0; i < n; ++i)
            if (ipiv[i] != i)
                std::swap(x[i], x[ipiv[i]]);

        // Forward substitution
        // ... (same as single RHS)

        // Back substitution
        // ... (same as single RHS)
    }
}
```

Parallelization over RHS is **embarrassingly parallel** -- no synchronization needed.

---

## 7. GPU Implementation

### 7.1 Batched Banded Solver

For GPU efficiency, we solve many independent banded systems in parallel:

```cuda
__global__ void banded_lu_batched(
    real* AB,        // Packed band matrices (ldab x n x batch_size)
    idx* ipiv,       // Pivot arrays (n x batch_size)
    idx n, idx kl, idx ku, idx ldab, idx batch_size)
{
    idx batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;

    // Each thread handles one complete system
    real* ab = AB + batch_idx * ldab * n;
    idx* piv = ipiv + batch_idx * n;

    idx kv = kl + ku;

    for (idx j = 0; j < n; ++j) {
        // Find pivot (sequential within thread)
        idx pivot = j;
        real max_val = fabs(ab[kv + j * ldab]);
        for (idx i = j + 1; i <= min(j + kl, n - 1); ++i) {
            real val = fabs(ab[kv + i - j + j * ldab]);
            if (val > max_val) {
                max_val = val;
                pivot = i;
            }
        }
        piv[j] = pivot;

        // Swap and eliminate (same as CPU, but within thread)
        // ...
    }
}
```

### 7.2 When GPU is Beneficial

| Scenario | CPU | GPU | Winner |
|----------|-----|-----|--------|
| Single small system (\f$n < 1000\f$) | 10 mus | 50 mus (launch overhead) | CPU |
| Single large system (\f$n > 10000\f$) | 1 ms | 0.5 ms | GPU |
| 1000 small systems | 10 ms | 0.1 ms | **GPU** |
| 1000 large systems | 1 s | 50 ms | **GPU** |

**GPU excels at batched solves**, which is exactly the pattern in radiative transfer (one system per spectral band or grid column).

---

## 8. Application: Radiative Transfer

### 8.1 Two-Stream Approximation

The radiative transfer equation for diffuse radiation:

\f[\mu \frac{dI}{d\tau} = I - S\f]

In the two-stream approximation (up/down fluxes \f$F^+\f$, \f$F^-\f$), discretized over \f$N\f$ atmospheric layers:

\f[\begin{pmatrix}
T_1 & R_1 & & \\
R_1 & T_1 & R_2 & \\
& \ddots & \ddots & \ddots \\
& & R_{N-1} & T_N
\end{pmatrix}
\begin{pmatrix} F^+_1 \\ F^-_1 \\ \vdots \\ F^-_N \end{pmatrix}
=
\begin{pmatrix} S_1 \\ S_2 \\ \vdots \\ S_N \end{pmatrix}\f]

where:
- \f$T_i\f$ = transmission matrix for layer \f$i\f$
- \f$R_i\f$ = reflection matrix coupling adjacent layers
- \f$S_i\f$ = source terms (direct solar, thermal emission)

**Matrix structure**: Block tridiagonal with \f$2 \times 2\f$ blocks -> banded with \f$k_l = k_u = 2\f$.

### 8.2 TUVX Photolysis Rates

TUVX (Tropospheric Ultraviolet-Visible) computes photolysis rate coefficients:

\f[J_i = \int_\lambda \sigma_i(\lambda) \phi_i(\lambda) F(\lambda) \, d\lambda\f]

where \f$F(\lambda)\f$ is the actinic flux from the radiative transfer solve.

**Computational pattern**:
1. For each wavelength band (~150 bands)
2. For each atmospheric column (millions in global models)
3. Solve tridiagonal/pentadiagonal system for fluxes

**This is ideal for batched banded solvers!**

### 8.3 Performance Considerations for NCAR Derecho

NCAR's Derecho supercomputer specifications:
- CPU: AMD EPYC Milan (128 cores/node)
- GPU: NVIDIA A100 (if using GPU partition)
- Memory bandwidth: ~400 GB/s per node

**Optimization for Derecho**:

```bash
# Build with AMD-specific optimizations
cmake .. -DCMAKE_CXX_FLAGS="-O3 -march=znver3 -fopenmp -ffast-math"
```

Key flags:
- `-march=znver3`: Zen 3 (Milan) specific instructions including AVX2
- `-fopenmp`: Enable OpenMP for multi-RHS parallelization
- `-ffast-math`: Allow reordering for SIMD (safe for banded solvers)

**Expected performance**:
- Tridiagonal (\f$n=100\f$ layers): ~0.5 mus per system
- Pentadiagonal (\f$n=100\f$ layers): ~1 mus per system
- Throughput: ~2M systems/second per core

---

## 9. Summary and Best Practices

### 9.1 Algorithm Selection Guide

| System Type | Recommended Solver |
|-------------|-------------------|
| Tridiagonal, single | Thomas algorithm |
| Tridiagonal, batched | GPU batched Thomas |
| General banded, single | `banded_solve()` |
| General banded, reuse matrix | `banded_lu()` + `banded_lu_solve()` |
| General banded, multi-RHS | `banded_lu_solve_multi()` |
| Very large single system | Consider iterative (CG with banded preconditioner) |

### 9.2 Numerical Robustness Checklist

1. **Check for diagonal dominance** -- if satisfied, pivoting is unnecessary
2. **Estimate condition number** -- use `banded_rcond()` after factorization
3. **Monitor pivot size** -- small pivots indicate near-singularity
4. **Use iterative refinement** for ill-conditioned systems

### 9.3 Performance Optimization Checklist

1. **Factor once, solve many** -- amortize factorization cost
2. **Batch independent systems** -- exploit parallelism
3. **Align memory** -- 64-byte alignment for AVX-512
4. **Profile memory bandwidth** -- banded solvers are often memory-bound
5. **Consider mixed precision** -- factor in double, solve in single if accuracy permits

### 9.4 Code References

| Function | Purpose | Complexity |
|----------|---------|------------|
| `BandedMatrix(n, kl, ku)` | Construct band storage | \f$O(n \cdot \text{ldab})\f$ |
| `banded_lu(A, ipiv)` | LU factorization | \f$O(n \cdot k_l \cdot (k_l + k_u))\f$ |
| `banded_lu_solve(A, ipiv, b)` | Solve with factored matrix | \f$O(n \cdot (k_l + k_u))\f$ |
| `banded_lu_solve_multi(A, ipiv, B, nrhs)` | Multiple RHS solve | \f$O(n \cdot (k_l + k_u) \cdot \text{nrhs})\f$ |
| `banded_solve(A, b, x)` | One-shot solve | Factor + Solve |
| `banded_matvec(A, x, y)` | Matrix-vector product | \f$O(n \cdot (k_l + k_u))\f$ |
| `banded_norm1(A)` | 1-norm | \f$O(n \cdot (k_l + k_u))\f$ |
| `banded_rcond(A, ipiv, anorm)` | Condition estimate | \f$O(n \cdot (k_l + k_u))\f$ |

---

## Appendix A: Fill-in Bound

**Claim**: With partial pivoting, \f$U\f$ has at most \f$k_l + k_u\f$ super-diagonals.

**Proof**:

Consider eliminating column \f$j\f$. The pivot row \f$p\f$ satisfies \f$j \leq p \leq j + k_l\f$.

After swapping row \f$p\f$ into row \f$j\f$:
- Original row \f$p\f$ had nonzeros in columns \f$\max(0, p - k_l)\f$ to \f$\min(n-1, p + k_u)\f$
- The rightmost nonzero is at column \f$p + k_u \leq (j + k_l) + k_u = j + k_l + k_u\f$

Since we're in row \f$j\f$ (the pivot row), this creates fill-in up to column \f$j + k_l + k_u\f$.

Thus \f$U\f$ has upper bandwidth \f$k_l + k_u\f$. \f$\square\f$

---

## Appendix B: Diagonal Dominance Preserved Under Elimination

**Claim**: If \f$A\f$ is strictly diagonally dominant, so is the Schur complement during elimination.

**Proof**:

After eliminating column 0, the \f$(1,1)\f$ element of the Schur complement is:
\f[\tilde{a}_{11} = a_{11} - \frac{a_{10}}{a_{00}} a_{01}\f]

The diagonal dominance of row 1 gives:
\f[|a_{11}| > |a_{10}| + |a_{12}| + \cdots\f]

We need to show \f$|\tilde{a}_{11}| > |\tilde{a}_{12}| + \cdots\f$ where \f$\tilde{a}_{1k} = a_{1k} - \frac{a_{10}}{a_{00}} a_{0k}\f$.

Using triangle inequality and the fact that \f$|a_{10}/a_{00}| < 1\f$ (from diagonal dominance of row 0):

\f[|\tilde{a}_{11}| \geq |a_{11}| - \left|\frac{a_{10}}{a_{00}}\right| |a_{01}|\f]

and

\f[|\tilde{a}_{1k}| \leq |a_{1k}| + \left|\frac{a_{10}}{a_{00}}\right| |a_{0k}|\f]

Summing over \f$k \neq 1\f$ and using diagonal dominance of rows 0 and 1:

\f[\sum_{k \neq 1} |\tilde{a}_{1k}| < \sum_{k \neq 1} |a_{1k}| + \left|\frac{a_{10}}{a_{00}}\right| \sum_{k \neq 0} |a_{0k}|\f]
\f[< |a_{11}| - |a_{10}| + \left|\frac{a_{10}}{a_{00}}\right| \cdot |a_{00}|\f]
\f[= |a_{11}|\f]

But we also have \f$|\tilde{a}_{11}| > |a_{11}| - |a_{10}||a_{01}|/|a_{00}| > |a_{11}| - |a_{10}|\f$ since \f$|a_{01}| < |a_{00}|\f$.

Combining these bounds shows \f$|\tilde{a}_{11}| > \sum_{k \neq 1} |\tilde{a}_{1k}|\f$. \f$\square\f$

This explains why Thomas algorithm (no pivoting) is stable for diagonally dominant tridiagonal systems.
