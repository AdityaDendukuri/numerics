# Week 4: Linear Solvers -- Theory and Parallel Implementation {#page_week4}

## 1. Introduction

Solving linear systems \f$A\mathbf{x} = \mathbf{b}\f$ is the computational backbone of scientific computing. This document provides a rigorous treatment of two solver methods and their parallel implementations:

1. **Conjugate Gradient (CG)** -- Krylov subspace method for SPD systems
2. **Thomas Algorithm** -- Direct solver for tridiagonal systems

---

## 2. Conjugate Gradient Method

### 2.1 Mathematical Foundation

#### The Minimization Perspective

For a symmetric positive definite (SPD) matrix \f$A \in \mathbb{R}^{n \times n}\f$, solving \f$A\mathbf{x} = \mathbf{b}\f$ is equivalent to minimizing the quadratic functional:

\f[\phi(\mathbf{x}) = \frac{1}{2}\mathbf{x}^T A \mathbf{x} - \mathbf{b}^T \mathbf{x}\f]

**Proof**: Taking the gradient:
\f[\nabla \phi(\mathbf{x}) = A\mathbf{x} - \mathbf{b}\f]

Setting \f$\nabla \phi = \mathbf{0}\f$ yields \f$A\mathbf{x} = \mathbf{b}\f$. Since \f$A\f$ is SPD, the Hessian \f$\nabla^2 \phi = A\f$ is positive definite, confirming this is a minimum. \f$\square\f$

The **residual** \f$\mathbf{r} = \mathbf{b} - A\mathbf{x}\f$ equals \f$-\nabla\phi\f$, pointing toward the minimum.

#### A-Conjugacy and Optimality

Two vectors \f$\mathbf{p}\f$ and \f$\mathbf{q}\f$ are **A-conjugate** (or A-orthogonal) if:
\f[\mathbf{p}^T A \mathbf{q} = 0\f]

**Key Theorem**: Given \f$n\f$ mutually A-conjugate directions \f$\{\mathbf{p}_0, \mathbf{p}_1, \ldots, \mathbf{p}_{n-1}\}\f$, the exact solution can be found in \f$n\f$ steps by sequential line minimization along each direction.

**Proof**: The A-conjugate vectors form a basis for \f$\mathbb{R}^n\f$. Express the error:
\f[\mathbf{x}^* - \mathbf{x}_0 = \sum_{i=0}^{n-1} \alpha_i \mathbf{p}_i\f]

Multiplying both sides by \f$\mathbf{p}_j^T A\f$:
\f[\mathbf{p}_j^T A (\mathbf{x}^* - \mathbf{x}_0) = \alpha_j \mathbf{p}_j^T A \mathbf{p}_j\f]

Thus \f$\alpha_j = \frac{\mathbf{p}_j^T A (\mathbf{x}^* - \mathbf{x}_0)}{\mathbf{p}_j^T A \mathbf{p}_j} = \frac{\mathbf{p}_j^T \mathbf{r}_0}{\mathbf{p}_j^T A \mathbf{p}_j}\f$

where \f$\mathbf{r}_0 = \mathbf{b} - A\mathbf{x}_0 = A(\mathbf{x}^* - \mathbf{x}_0)\f$. \f$\square\f$

#### Krylov Subspace Connection

The CG method generates A-conjugate directions from the **Krylov subspace**:
\f[\mathcal{K}_k(A, \mathbf{r}_0) = \text{span}\{\mathbf{r}_0, A\mathbf{r}_0, A^2\mathbf{r}_0, \ldots, A^{k-1}\mathbf{r}_0\}\f]

After \f$k\f$ iterations, CG finds:
\f[\mathbf{x}_k = \arg\min_{\mathbf{x} \in \mathbf{x}_0 + \mathcal{K}_k} \|\mathbf{x} - \mathbf{x}^*\|_A\f]

where \f$\|\mathbf{v}\|_A = \sqrt{\mathbf{v}^T A \mathbf{v}}\f$ is the **A-norm** (energy norm).

### 2.2 Algorithm

Starting from \f$\mathbf{x}_0\f$ with residual \f$\mathbf{r}_0 = \mathbf{b} - A\mathbf{x}_0\f$, set \f$\mathbf{p}_0 = \mathbf{r}_0\f$.

**Step size** (exact line minimization): Minimize \f$\phi(\mathbf{x}_k + \alpha \mathbf{p}_k)\f$:
\f[\frac{d}{d\alpha}\phi(\mathbf{x}_k + \alpha \mathbf{p}_k) = \mathbf{p}_k^T(A\mathbf{x}_k + \alpha A\mathbf{p}_k - \mathbf{b}) = 0\f]
\f[\alpha_k = \frac{\mathbf{p}_k^T \mathbf{r}_k}{\mathbf{p}_k^T A \mathbf{p}_k}\f]

**Update**:
\f[\mathbf{x}_{k+1} = \mathbf{x}_k + \alpha_k \mathbf{p}_k\f]
\f[\mathbf{r}_{k+1} = \mathbf{r}_k - \alpha_k A \mathbf{p}_k\f]

**New search direction** (enforce A-conjugacy with all previous directions):
\f[\mathbf{p}_{k+1} = \mathbf{r}_{k+1} + \beta_k \mathbf{p}_k\f]

where \f$\beta_k = \frac{\mathbf{r}_{k+1}^T \mathbf{r}_{k+1}}{\mathbf{r}_k^T \mathbf{r}_k}\f$

**Remarkable property**: Due to the Krylov structure, we only need the previous direction--not all previous directions--to maintain A-conjugacy.

### 2.3 The CG Algorithm

```
Input: A (SPD), b, x_0, tol, max_iter
Output: x (approximate solution)

r = b - A*x
p = r
rho_old = rTr

for k = 0, 1, 2, ... until convergence:
    q = A*p                    // Matrix-vector product
    alpha = rho_old / (pTq)         // Step length
    x = x + alpha*p               // Update solution
    r = r - alpha*q               // Update residual
    rho_new = rTr               // New residual norm squared

    if sqrtrho_new < tol: break    // Convergence check

    beta = rho_new / rho_old         // Improvement ratio
    p = r + beta*p               // New search direction
    rho_old = rho_new
```

### 2.4 Convergence Analysis

**Theorem** (CG Convergence Rate): After \f$k\f$ iterations:
\f[\frac{\|\mathbf{x}_k - \mathbf{x}^*\|_A}{\|\mathbf{x}_0 - \mathbf{x}^*\|_A} \leq 2\left(\frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}\right)^k\f]

where \f$\kappa = \lambda_{\max}/\lambda_{\min}\f$ is the **condition number** of \f$A\f$.

**Proof sketch**: CG minimizes over polynomials \f$p_k\f$ of degree \f$k\f$ with \f$p_k(0) = 1\f$:
\f[\|\mathbf{x}_k - \mathbf{x}^*\|_A = \min_{p_k} \|p_k(A)(\mathbf{x}_0 - \mathbf{x}^*)\|_A\f]

Using Chebyshev polynomials (optimal for minimizing \f$\max_{\lambda \in [\lambda_{\min}, \lambda_{\max}]} |p_k(\lambda)|\f$) yields the bound. \f$\square\f$

**Practical implications**:
| Condition Number | Iterations for 6 digits |
|-----------------|------------------------|
| \f$\kappa = 10\f$ | ~7 |
| \f$\kappa = 100\f$ | ~23 |
| \f$\kappa = 10^4\f$ | ~230 |
| \f$\kappa = 10^6\f$ | ~2300 |

### 2.5 Operation Count

Per iteration:
| Operation | FLOPs | Memory Access |
|-----------|-------|---------------|
| \f$\mathbf{q} = A\mathbf{p}\f$ | \f$2n^2\f$ (dense) or \f$O(nnz)\f$ (sparse) | \f$O(n^2)\f$ or \f$O(nnz)\f$ |
| \f$\alpha = \rho/(\mathbf{p}^T\mathbf{q})\f$ | \f$2n\f$ | \f$2n\f$ reads |
| \f$\mathbf{x} = \mathbf{x} + \alpha\mathbf{p}\f$ | \f$2n\f$ | \f$3n\f$ |
| \f$\mathbf{r} = \mathbf{r} - \alpha\mathbf{q}\f$ | \f$2n\f$ | \f$3n\f$ |
| \f$\rho = \mathbf{r}^T\mathbf{r}\f$ | \f$2n\f$ | \f$n\f$ reads |
| \f$\mathbf{p} = \mathbf{r} + \beta\mathbf{p}\f$ | \f$2n\f$ | \f$3n\f$ |

**Total per iteration**: \f$2n^2 + 10n\f$ FLOPs (dense), dominated by matvec.

---

## 3. Thomas Algorithm (Tridiagonal Solver)

### 3.1 Problem Structure

A **tridiagonal system** has the form:

\f[\begin{pmatrix}
b_0 & c_0 \\
a_0 & b_1 & c_1 \\
& a_1 & b_2 & c_2 \\
& & \ddots & \ddots & \ddots \\
& & & a_{n-3} & b_{n-2} & c_{n-2} \\
& & & & a_{n-2} & b_{n-1}
\end{pmatrix}
\begin{pmatrix} x_0 \\ x_1 \\ x_2 \\ \vdots \\ x_{n-2} \\ x_{n-1} \end{pmatrix}
=
\begin{pmatrix} d_0 \\ d_1 \\ d_2 \\ \vdots \\ d_{n-2} \\ d_{n-1} \end{pmatrix}\f]

**Storage**:
- \f$\mathbf{a}\f$: lower diagonal, size \f$n-1\f$ (elements \f$a_0, \ldots, a_{n-2}\f$)
- \f$\mathbf{b}\f$: main diagonal, size \f$n\f$
- \f$\mathbf{c}\f$: upper diagonal, size \f$n-1\f$ (elements \f$c_0, \ldots, c_{n-2}\f$)

### 3.2 Thomas Algorithm via LU Structure

The Thomas algorithm is Gaussian elimination specialized for tridiagonal structure.

**LU Decomposition**: Factor \f$A = LU\f$ where:

\f[L = \begin{pmatrix}
1 \\
l_0 & 1 \\
& l_1 & 1 \\
& & \ddots & \ddots \\
& & & l_{n-2} & 1
\end{pmatrix}, \quad
U = \begin{pmatrix}
u_0 & c_0 \\
& u_1 & c_1 \\
& & u_2 & c_2 \\
& & & \ddots & \ddots \\
& & & & u_{n-1}
\end{pmatrix}\f]

**Deriving the factors**: From \f$A = LU\f$:

Row 0: \f$b_0 = u_0\f$, so \f$u_0 = b_0\f$

Row \f$i\f$ (\f$i \geq 1\f$):
- Position \f$(i, i-1)\f$: \f$a_{i-1} = l_{i-1} \cdot u_{i-1}\f$, so \f$l_{i-1} = a_{i-1}/u_{i-1}\f$
- Position \f$(i, i)\f$: \f$b_i = l_{i-1} \cdot c_{i-1} + u_i\f$, so \f$u_i = b_i - l_{i-1} \cdot c_{i-1}\f$

**Forward elimination** (computing \f$L^{-1}\mathbf{d}\f$):

Let \f$\mathbf{y} = U\mathbf{x}\f$, solve \f$L\mathbf{y} = \mathbf{d}\f$:
\f[y_0 = d_0\f]
\f[y_i = d_i - l_{i-1} \cdot y_{i-1} \quad \text{for } i = 1, \ldots, n-1\f]

**Back substitution** (solving \f$U\mathbf{x} = \mathbf{y}\f$):
\f[x_{n-1} = y_{n-1} / u_{n-1}\f]
\f[x_i = (y_i - c_i \cdot x_{i+1}) / u_i \quad \text{for } i = n-2, \ldots, 0\f]

### 3.3 The Thomas Algorithm

Combining and simplifying (using \f$\mathbf{b}\f$ and \f$\mathbf{d}\f$ as working storage):

```
Input: a[0..n-2], b[0..n-1], c[0..n-2], d[0..n-1]
Output: x[0..n-1]

// Create working copies
b_work = copy(b)
d_work = copy(d)

// Forward elimination
for i = 1 to n-1:
    w = a[i-1] / b_work[i-1]
    b_work[i] = b_work[i] - w * c[i-1]
    d_work[i] = d_work[i] - w * d_work[i-1]

// Back substitution
x[n-1] = d_work[n-1] / b_work[n-1]
for i = n-2 down to 0:
    x[i] = (d_work[i] - c[i] * x[i+1]) / b_work[i]
```

### 3.4 Complexity Analysis

**Forward elimination**:
- Loop iterations: \f$n-1\f$
- Per iteration: 1 division, 2 multiplications, 2 subtractions = 5 FLOPs
- Total: \f$5(n-1)\f$ FLOPs

**Back substitution**:
- First element: 1 division
- Remaining \f$n-1\f$ elements: 1 subtraction, 1 multiplication, 1 division = 3 FLOPs each
- Total: \f$1 + 3(n-1) = 3n - 2\f$ FLOPs

**Grand total**: \f$5(n-1) + 3n - 2 = 8n - 7\f$ FLOPs = \f$O(n)\f$

This is **optimal**--any algorithm must at least read the \f$O(n)\f$ input values.

### 3.5 Stability Analysis

**Theorem**: The Thomas algorithm is stable if \f$A\f$ is **diagonally dominant**:
\f[|b_i| > |a_{i-1}| + |c_i|, \quad i = 0, \ldots, n-1\f]

(with \f$a_{-1} = c_{n-1} = 0\f$).

**Proof**: Under diagonal dominance, \f$|w| = |a_{i-1}/b_{\text{work},i-1}| < 1\f$, so:
\f[|b_{\text{work},i}| = |b_i - w \cdot c_{i-1}| \geq |b_i| - |w||c_{i-1}| > |c_i|\f]

Thus modified diagonal elements remain dominant and bounded away from zero. \f$\square\f$

**Example** (1D Laplacian): The system with \f$a_i = c_i = -1\f$, \f$b_i = 2\f$ satisfies \f$|2| > |-1| + |-1|\f$, so Thomas is stable.

---

## 4. CUDA Implementation

### 4.1 CG on GPU

The CG algorithm maps naturally to GPU by parallelizing each operation:

```cpp
SolverResult cg(const Matrix& A, const Vector& b, Vector& x,
                real tol, idx max_iter, Exec exec) {

    Vector r(n), p(n), Ap(n);

    if (exec == Exec::gpu) {
        // Transfer data to GPU
        A.to_gpu(); b.to_gpu(); x.to_gpu();
        r.to_gpu(); p.to_gpu(); Ap.to_gpu();
    }

    // r = b - A*x
    matvec(A, x, r, exec);           // GPU: parallel matvec kernel
    // r = b - r (combined via axpy)
    scale(r, -1.0, exec);            // GPU: parallel scale kernel
    axpy(1.0, b, r, exec);           // GPU: parallel axpy kernel

    // p = r (GPU: device-to-device copy)

    real rsold = dot(r, r, exec);    // GPU: parallel reduction

    for (idx iter = 0; iter < max_iter; ++iter) {
        matvec(A, p, Ap, exec);      // Dominant cost

        real pAp = dot(p, Ap, exec);
        real alpha = rsold / pAp;

        axpy(alpha, p, x, exec);     // x += alpha*p
        axpy(-alpha, Ap, r, exec);   // r -= alpha*Ap

        real rsnew = dot(r, r, exec);
        if (sqrt(rsnew) < tol) break;

        real beta = rsnew / rsold;
        scale(p, beta, exec);        // p = beta*p
        axpy(1.0, r, p, exec);       // p += r

        rsold = rsnew;
    }

    if (exec == Exec::gpu) x.to_cpu();
    return result;
}
```

#### GPU Kernels Used

**Parallel dot product** (reduction):
```cuda
__global__ void k_dot(const real* x, const real* y, real* result, idx n) {
    __shared__ real sdata[BLOCK_SIZE];
    idx tid = threadIdx.x;
    idx i = blockIdx.x * blockDim.x + tid;

    // Each thread computes partial product
    sdata[tid] = (i < n) ? x[i] * y[i] : 0;
    __syncthreads();

    // Tree reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Block result added atomically to global result
    if (tid == 0) atomicAdd(result, sdata[0]);
}
```

**Parallel AXPY** (\f$\mathbf{y} = \alpha\mathbf{x} + \mathbf{y}\f$):
```cuda
__global__ void k_axpy(real alpha, const real* x, real* y, idx n) {
    idx i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += alpha * x[i];
}
```

**Parallel matrix-vector product**:
```cuda
__global__ void k_matvec(const real* A, const real* x, real* y,
                         idx rows, idx cols) {
    idx i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows) {
        real sum = 0;
        for (idx j = 0; j < cols; ++j)
            sum += A[i * cols + j] * x[j];
        y[i] = sum;
    }
}
```

#### Performance Considerations

| Problem Size | CPU Wins | GPU Wins |
|-------------|----------|----------|
| \f$n < 500\f$ | ok (kernel overhead) | |
| \f$n > 1000\f$ | | ok (parallelism) |

The crossover point depends on:
- GPU memory bandwidth
- Kernel launch latency (~5-20 mus)
- Number of CG iterations

### 4.2 Thomas Algorithm on GPU

The Thomas algorithm is inherently **sequential** (forward sweep depends on previous row). However, we can parallelize across **multiple independent systems**.

#### Batched Thomas Kernel

When solving \f$k\f$ independent tridiagonal systems (common in ADI methods):

```cuda
__global__ void k_thomas_batched(
    const real* a, const real* b, const real* c,
    const real* d, real* x,
    real* b_work, real* d_work,  // Workspace
    idx n, idx batch_size)
{
    // Each thread handles one complete system
    idx batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;

    // Offset into packed arrays
    idx off_sub = batch_idx * (n - 1);   // For a, c
    idx off_main = batch_idx * n;         // For b, d, x

    // Copy to workspace
    for (idx i = 0; i < n; ++i) {
        b_work[off_main + i] = b[off_main + i];
        d_work[off_main + i] = d[off_main + i];
    }

    // Forward elimination (sequential within thread)
    for (idx i = 1; i < n; ++i) {
        real w = a[off_sub + i - 1] / b_work[off_main + i - 1];
        b_work[off_main + i] -= w * c[off_sub + i - 1];
        d_work[off_main + i] -= w * d_work[off_main + i - 1];
    }

    // Back substitution (sequential within thread)
    x[off_main + n - 1] = d_work[off_main + n - 1] / b_work[off_main + n - 1];
    for (idx i = n - 1; i > 0; --i) {
        x[off_main + i - 1] = (d_work[off_main + i - 1]
            - c[off_sub + i - 1] * x[off_main + i]) / b_work[off_main + i - 1];
    }
}
```

**Wrapper function**:
```cpp
void thomas_batched(const real* a, const real* b, const real* c,
                    const real* d, real* x, idx n, idx batch_size) {
    real *b_work, *d_work;
    cudaMalloc(&b_work, batch_size * n * sizeof(real));
    cudaMalloc(&d_work, batch_size * n * sizeof(real));

    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;

    k_thomas_batched<<<blocks, threads>>>(
        a, b, c, d, x, b_work, d_work, n, batch_size);

    cudaFree(b_work);
    cudaFree(d_work);
}
```

#### When Batched Thomas Helps

| Scenario | Recommendation |
|----------|---------------|
| Single system | Use CPU Thomas |
| 100+ independent systems | Use GPU batched |
| ADI with \f$n_y \times n_z\f$ pencils | GPU batched excels |

---

## 5. MPI Implementation (Distributed Memory)

### 5.1 Distributed CG

For large systems that don't fit in single-node memory, distribute the matrix and vectors across \f$P\f$ MPI ranks.

#### Data Distribution

**1D row distribution**: Rank \f$p\f$ owns rows \f$[p \cdot (n/P), (p+1) \cdot (n/P))\f$

```
Rank 0: rows 0 to n/P - 1
Rank 1: rows n/P to 2n/P - 1
...
Rank P-1: rows (P-1)n/P to n - 1
```

Each rank stores:
- Local portion of \f$A\f$: size \f$(n/P) \times n\f$ (full rows)
- Local portions of \f$\mathbf{x}, \mathbf{b}, \mathbf{r}, \mathbf{p}\f$: size \f$n/P\f$

#### Distributed Operations

**Matrix-vector product** (\f$\mathbf{y} = A\mathbf{x}\f$):

```cpp
void distributed_matvec(const Matrix& A_local, const Vector& x_local,
                        Vector& y_local, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    idx local_rows = A_local.rows();
    idx n = local_rows * size;  // Total size

    // Gather full x vector (each rank needs all of x)
    Vector x_full(n);
    MPI_Allgather(x_local.data(), local_rows, MPI_DOUBLE,
                  x_full.data(), local_rows, MPI_DOUBLE, comm);

    // Local matvec (each rank computes its rows)
    for (idx i = 0; i < local_rows; ++i) {
        real sum = 0;
        for (idx j = 0; j < n; ++j)
            sum += A_local(i, j) * x_full[j];
        y_local[i] = sum;
    }
}
```

**Distributed dot product**:

```cpp
real distributed_dot(const Vector& x_local, const Vector& y_local,
                     MPI_Comm comm) {
    // Local partial sum
    real local_sum = 0;
    for (idx i = 0; i < x_local.size(); ++i)
        local_sum += x_local[i] * y_local[i];

    // Global reduction
    real global_sum;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, comm);

    return global_sum;
}
```

**Distributed AXPY** (no communication needed):

```cpp
void distributed_axpy(real alpha, const Vector& x_local, Vector& y_local) {
    for (idx i = 0; i < x_local.size(); ++i)
        y_local[i] += alpha * x_local[i];
}
```

#### Distributed CG Algorithm

```cpp
SolverResult distributed_cg(const Matrix& A_local, const Vector& b_local,
                            Vector& x_local, real tol, idx max_iter,
                            MPI_Comm comm) {
    idx local_n = b_local.size();
    Vector r_local(local_n), p_local(local_n), Ap_local(local_n);

    // r = b - A*x
    distributed_matvec(A_local, x_local, r_local, comm);
    for (idx i = 0; i < local_n; ++i)
        r_local[i] = b_local[i] - r_local[i];

    // p = r
    p_local = r_local;

    real rsold = distributed_dot(r_local, r_local, comm);

    for (idx iter = 0; iter < max_iter; ++iter) {
        distributed_matvec(A_local, p_local, Ap_local, comm);

        real pAp = distributed_dot(p_local, Ap_local, comm);
        real alpha = rsold / pAp;

        distributed_axpy(alpha, p_local, x_local);
        distributed_axpy(-alpha, Ap_local, r_local);

        real rsnew = distributed_dot(r_local, r_local, comm);

        if (sqrt(rsnew) < tol) {
            return {iter + 1, sqrt(rsnew), true};
        }

        real beta = rsnew / rsold;
        for (idx i = 0; i < local_n; ++i)
            p_local[i] = r_local[i] + beta * p_local[i];

        rsold = rsnew;
    }

    return {max_iter, sqrt(rsold), false};
}
```

#### Communication Analysis

Per CG iteration:
| Operation | Communication |
|-----------|--------------|
| `matvec` | `MPI_Allgather` -- \f$O(n)\f$ data, \f$O(\log P)\f$ latency |
| `dot` (x2) | `MPI_Allreduce` -- \f$O(1)\f$ data, \f$O(\log P)\f$ latency |
| `axpy` (x3) | None |

**Total per iteration**:
- 1 Allgather: \f$O(n/P \cdot P) = O(n)\f$ bytes
- 2 Allreduce: \f$O(1)\f$ bytes each

Communication is dominated by the Allgather in matvec.

### 5.2 Distributed Thomas (Pipeline)

For a single tridiagonal system distributed across ranks, use **pipelining**:

```
Rank 0: rows 0 to n/P - 1
Rank 1: rows n/P to 2n/P - 1  (needs data from Rank 0)
...
```

**Forward sweep**: Each rank waits for modified values from previous rank:

```cpp
void distributed_thomas_forward(Vector& b_local, Vector& d_local,
                                const Vector& a_local, const Vector& c_local,
                                MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    idx local_n = b_local.size();

    // Receive boundary data from previous rank
    if (rank > 0) {
        real recv_b, recv_d;
        MPI_Recv(&recv_b, 1, MPI_DOUBLE, rank - 1, 0, comm, MPI_STATUS_IGNORE);
        MPI_Recv(&recv_d, 1, MPI_DOUBLE, rank - 1, 1, comm, MPI_STATUS_IGNORE);

        // First local row depends on last row of previous rank
        real w = a_local[0] / recv_b;  // a_local[0] connects to previous rank
        b_local[0] -= w * c_local[local_n - 1];  // Approximate; actual indexing depends on layout
        d_local[0] -= w * recv_d;
    }

    // Local forward elimination
    for (idx i = 1; i < local_n; ++i) {
        real w = a_local[i] / b_local[i - 1];
        b_local[i] -= w * c_local[i - 1];
        d_local[i] -= w * d_local[i - 1];
    }

    // Send boundary data to next rank
    if (rank < size - 1) {
        MPI_Send(&b_local[local_n - 1], 1, MPI_DOUBLE, rank + 1, 0, comm);
        MPI_Send(&d_local[local_n - 1], 1, MPI_DOUBLE, rank + 1, 1, comm);
    }
}
```

**Note**: Pipeline parallelism has limited scalability. For better parallel efficiency, consider:
- **Cyclic Reduction**: \f$O(\log n)\f$ parallel depth
- **Spike Algorithm**: Domain decomposition with coupling solve

---

## 6. Hybrid MPI+CUDA Implementation

For multi-GPU clusters, combine MPI (inter-node) with CUDA (intra-node):

```cpp
SolverResult hybrid_cg(const Matrix& A_local, const Vector& b_local,
                       Vector& x_local, real tol, idx max_iter,
                       MPI_Comm comm) {
    // Each MPI rank has one GPU
    int rank;
    MPI_Comm_rank(comm, &rank);
    cudaSetDevice(rank % num_gpus_per_node);

    // Transfer local data to GPU
    A_local.to_gpu();
    b_local.to_gpu();
    x_local.to_gpu();

    // ... CG iterations using:
    // - GPU kernels for local operations
    // - CUDA-aware MPI for communication

    // Example: GPU-aware Allgather
    MPI_Allgather(x_local.gpu_data(), local_n, MPI_DOUBLE,
                  x_full_gpu, local_n, MPI_DOUBLE, comm);
}
```

**CUDA-aware MPI** allows direct GPU-to-GPU transfers without staging through CPU memory.

---

## 7. Summary

| Method | Complexity | Parallelization | Best Use Case |
|--------|-----------|-----------------|---------------|
| CG (CPU) | \f$O(n^2)\f$/iter | OpenMP threads | Small-medium SPD |
| CG (GPU) | \f$O(n^2)\f$/iter | CUDA kernels | Large SPD (\f$n > 1000\f$) |
| CG (MPI) | \f$O(n^2/P)\f$/iter + comm | Distributed | Huge SPD |
| Thomas (CPU) | \f$O(n)\f$ | None (sequential) | Single tridiagonal |
| Thomas Batched (GPU) | \f$O(n \cdot k/P_{GPU})\f$ | Thread per system | Many tridiagonals |
| Thomas (MPI) | \f$O(n/P)\f$ + pipeline | Limited | Large single tridiag |

Summary:
1. CG parallelizes naturally; matvec dominates cost
2. Thomas is inherently sequential; parallelize across systems, not within
3. GPU advantage requires sufficient problem size to amortize launch overhead
4. MPI CG needs careful attention to communication (Allgather is expensive)
