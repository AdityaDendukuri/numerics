# Parallel and Distributed Computing {#page_parallel}

This page covers the three parallelism strategies used in this library: MPI for
distributed-memory clusters, CUDA for GPU acceleration, and banded matrix solvers
that exploit sparsity structure for \f$O(nk^2)\f$ factorization cost.  Each
section describes the programming model, the key performance tradeoffs, and the
library API.

---

## MPI: Distributed Memory {#sec_mpi}

### The SPMD Model

MPI (Message Passing Interface) follows the **SPMD** (Single Program, Multiple
Data) model: every process runs the same executable but works on a different
portion of the data.  Each process has completely private memory; no shared address
space exists.  All data exchange is **explicit** -- the programmer controls every
transfer.

This approach scales to thousands of nodes across high-speed interconnects because
it imposes no hardware constraint for cache coherence or memory sharing.

### Point-to-Point Communication

`MPI_Send` and `MPI_Recv` are the building blocks.  Their main hazard is
**deadlock**: if every process in a communicating pair calls `MPI_Recv` before
`MPI_Send`, all processes wait forever.  `MPI_Sendrecv` resolves this by
combining both operations atomically:

```cpp
MPI_Sendrecv(&send_data, 1, MPI_DOUBLE, partner, 0,
             &recv_data, 1, MPI_DOUBLE, partner, 0,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
```

**Non-blocking** variants (`MPI_Isend`, `MPI_Irecv`) return immediately and allow
computation to proceed while the message is in flight.  `MPI_Wait` blocks until
completion.  This is the foundation for overlapping communication with computation,
discussed below.

### Collective Operations

Collectives involve all processes in a communicator simultaneously.  They are
implemented with tree-based algorithms optimised by the MPI library:

| Collective | Description | Latency complexity |
|------------|-------------|-------------------|
| `MPI_Bcast` | Root sends data to all | \f$O(\log p)\f$ |
| `MPI_Reduce` | Combine all data to root | \f$O(\log p)\f$ |
| `MPI_Allreduce` | Reduce, result to all | \f$O(\log p)\f$ |
| `MPI_Gather` | Collect to root | \f$O(p)\f$ |
| `MPI_Scatter` | Distribute from root | \f$O(p)\f$ |
| `MPI_Allgather` | Gather, result to all | \f$O(p)\f$ |

Prefer `MPI_Allreduce` over `MPI_Reduce` + `MPI_Bcast` -- the combined collective
is typically faster due to fewer synchronisation barriers.

### Distributed Dot Product

Each rank holds a contiguous block of \f$\mathbf{x}\f$ and \f$\mathbf{y}\f$.
The global dot product is assembled via a local partial sum followed by an
`Allreduce`:

\f[
\mathbf{x}^T\mathbf{y} = \sum_r \sum_{j \in \text{local}_r} x_j y_j
\f]

The communication cost is one `Allreduce` -- \f$O(\log p)\f$ latency.

### Distributed Matrix-Vector Product

For \f$\mathbf{y} = A\mathbf{x}\f$ with A stored in 1-D row distribution:

1. **Allgather** the distributed vector \f$\mathbf{x}\f$ so every process holds
   the full vector (\f$O(n)\f$ per process).
2. Each process computes its local rows of \f$\mathbf{y}\f$ independently.

The communication cost is \f$O(n)\f$ per process per matvec.

### Parallel Conjugate Gradient

CG for \f$A\mathbf{x} = \mathbf{b}\f$ requires only matvec, dot products, and
AXPY operations.  In distributed form, each of \f$k\f$ iterations costs:

- 2 `Allreduce` calls (for \f$\rho_{\text{new}} = \mathbf{r}^T\mathbf{r}\f$ and
  \f$\alpha = \rho / \mathbf{p}^T A\mathbf{p}\f$)
- 1 `Allgather` (to gather \f$\mathbf{p}\f$ for the distributed matvec)

Total communication: \f$O(k \log p)\f$ latency across all iterations.

### Hiding Latency with Non-Blocking Collectives

MPI-3 introduced non-blocking collectives.  A typical pattern in parallel CG
issues the `Allreduce` for \f$\rho\f$ immediately after computing the local
partial sum, then performs independent local AXPY updates while the reduction
proceeds:

```cpp
MPI_Request req;
MPI_Iallreduce(&local_dot, &global_dot, 1, MPI_DOUBLE,
               MPI_SUM, comm, &req);

// Independent local work (AXPY, etc.) runs here in parallel
do_local_axpy_updates();

MPI_Wait(&req, MPI_STATUS_IGNORE);  // Synchronise before using global_dot
```

For strong-scaling regimes where communication latency dominates, this overlap
can recover 20-40 % of iteration time.

API: @ref num::mpi::dot, @ref num::mpi::norm, @ref num::mpi::allreduce_sum,
@ref num::mpi::broadcast

---

## CUDA: GPU Computing {#sec_cuda}

### GPU Hierarchy

NVIDIA GPUs expose a three-level hierarchy:

- **SMs (Streaming Multiprocessors)**: Independent processing units, each with
  its own registers, shared memory, and warp schedulers.
- **Warps**: Groups of 32 threads executing in **SIMT** (Single Instruction,
  Multiple Threads) lockstep.  All threads in a warp issue the same instruction
  each cycle; divergent branches serialise execution.
- **Threads**: Individual scalar lanes within a warp.

A kernel launch specifies a grid of thread blocks; each block is assigned to one
SM and its threads share that SM's shared memory.

### GPU Memory Hierarchy

| Memory type | Scope | Capacity | Approximate bandwidth | Lifetime |
|-------------|-------|----------|-----------------------|----------|
| Registers | Thread | ~255 per thread | -- (zero latency) | Thread |
| Shared / L1 | Block | 48-164 KB per SM | ~100 TB/s | Block |
| Global | Grid | GBs | ~500 GB/s | Application |
| Constant | Grid | 64 KB | cached | Application |

Shared memory bandwidth (~100 TB/s) is roughly 200x higher than global memory
bandwidth (~500 GB/s).  The central optimization strategy for GPU kernels is
to **load data from global memory into shared memory once and reuse it many
times** within a block.

### Coalesced Global Memory Access

When threads in a warp read consecutive addresses, the hardware merges the 32
individual requests into a single 128-byte transaction.  When threads read
strided addresses, each request becomes a separate transaction -- up to 32x more
bandwidth consumed.

```
Coalesced  (thread i reads element i):      1 transaction
Strided    (thread i reads element i*stride): up to 32 transactions
```

Row-major matrix layouts naturally produce coalesced row reads; column reads are
strided and should be avoided in the innermost loop or loaded via shared memory.

### Tiled Matrix Multiplication with Shared Memory

The naive GPU matmul reads every element of A and B \f$N\f$ times from global
memory.  With \f$T \times T\f$ tiles loaded into shared memory:

- Each \f$T \times T\f$ tile is loaded **once** into shared memory per tile
  iteration.
- It is reused \f$T\f$ times within the block for the partial dot products.
- **Global memory reads are reduced by a factor of \f$T\f$.**

For \f$T = 16\f$, a 1024x1024 multiplication reduces global memory traffic from
\f$\sim 16\f$ GB to \f$\sim 1\f$ GB.  The `__syncthreads()` barrier after loading
each tile ensures all threads see the complete tile before the compute phase
begins.

### Kernel Launch Overhead and Crossover Points

Launching a CUDA kernel incurs roughly 5-20 mus of overhead (PCIe transfer setup,
driver dispatch).  For small operands the launch cost dominates and the GPU never
wins.  Crossover points measured on this library's kernels:

| Operation | GPU crossover point |
|-----------|---------------------|
| Dot product | GPU never wins (memory-bound + launch overhead) |
| AXPY | ~64 K elements |
| Matvec | ~256x256 matrix |
| Matmul | GPU always wins (compute-bound) |

For production use, keep data on the GPU across multiple kernel calls to amortise
transfer costs.

### Batched Thomas Algorithm

The Thomas algorithm solves a tridiagonal system of size \f$n\f$ in \f$O(n)\f$
operations -- it is inherently sequential.  The GPU makes it profitable by running
**one thread per independent system**:

```
Thread 0: system 0 (one atmospheric column, one wavelength band)
Thread 1: system 1
...
Thread k-1: system k-1
```

For \f$k \geq 1024\f$ independent systems, warp occupancy is high and the kernel
achieves near-peak utilisation.  This is exactly the pattern arising in radiative
transfer codes (one tridiagonal per atmospheric column per spectral band) and in
banded matrix applications with large batch counts.

API: @ref num::cuda::matvec, @ref num::cuda::matmul,
@ref num::cuda::thomas_batched

---

## Banded Matrix Systems {#sec_banded}

### Band Structure

A matrix \f$A \in \mathbb{R}^{n \times n}\f$ is **banded** with lower bandwidth
\f$k_l\f$ and upper bandwidth \f$k_u\f$ if

\f[
A_{ij} = 0 \quad \text{for } |i - j| > k_l \text{ (lower) or } |i - j| > k_u \text{ (upper)}
\f]

Common special cases:

- **Tridiagonal**: \f$k_l = k_u = 1\f$ -- arises from second-order finite
  difference discretisations of 1-D boundary value problems.
- **Pentadiagonal**: \f$k_l = k_u = 2\f$ -- arises from fourth-order stencils
  and two-stream radiative transfer discretisations.

Exploiting band structure reduces storage from \f$O(n^2)\f$ to
\f$O(n(k_l + k_u + 1))\f$ and factorization cost from \f$O(n^3)\f$ to
\f$O(n k_l (k_l + k_u))\f$.

### LAPACK-Style Band Storage

The standard compact format stores diagonals in the rows of a 2-D array with
leading dimension \f$\text{ldab} = 2k_l + k_u + 1\f$.  Element \f$A_{ij}\f$
within the band maps to

\f[
\text{band}[k_l + k_u + i - j][j] = A_{ij}
\f]

using 0-based indexing.  The array has \f$\text{ldab}\f$ rows and \f$n\f$
columns.

The upper \f$k_l\f$ rows (indices \f$0 \ldots k_l - 1\f$) are **reserved for
fill-in** during LU factorisation.  With partial pivoting, the \f$U\f$ factor can
develop up to \f$k_l\f$ additional super-diagonals beyond the original \f$k_u\f$,
so the storage must accommodate upper bandwidth \f$k_l + k_u\f$.

Column-major storage ensures that all elements in a column are contiguous in
memory, giving sequential access patterns during forward and back substitution.

### Banded LU with Partial Pivoting

The factorisation \f$PA = LU\f$ proceeds column by column.  For column \f$j\f$:

1. Search rows \f$j\f$ to \f$\min(j + k_l,\, n-1)\f$ for the largest-magnitude
   pivot.
2. Swap the pivot row with row \f$j\f$.
3. Compute multipliers \f$l_{ij} = a_{ij}/a_{jj}\f$ for rows \f$j+1\f$ to
   \f$j+k_l\f$.
4. Update the sub-matrix for columns \f$j+1\f$ to \f$j+k_u\f$ (original upper
   bandwidth).

Fill-in reaches at most \f$k_l\f$ positions above the original upper diagonal,
so \f$U\f$ has bandwidth \f$k_l + k_u\f$ -- exactly the extra rows reserved in
the band array.

**Complexity comparison:**

| Method | Factorization | Solve | Storage |
|--------|--------------|-------|---------|
| Dense LU | \f$\tfrac{2}{3}n^3\f$ | \f$2n^2\f$ | \f$n^2\f$ |
| Banded LU | \f$O\!\left(n k_l (k_l + k_u)\right)\f$ | \f$O\!\left(n(k_l+k_u)\right)\f$ | \f$(2k_l+k_u+1)n\f$ |
| Thomas (tridiagonal) | \f$8n\f$ | \f$5n\f$ | \f$4n\f$ |

For \f$n = 10\,000\f$ with a pentadiagonal system (\f$k_l = k_u = 2\f$):

- Dense factorization: \f$\approx 6.7 \times 10^{11}\f$ FLOPs
- Banded factorization: \f$10\,000 \times 2 \times 4 = 80\,000\f$ FLOPs
- **Speedup: \f$\approx 8 \times 10^6\times\f$**

### Numerical Stability: Growth Factor

Partial pivoting bounds the growth factor for dense LU at \f$2^{n-1}\f$ in the
worst case.  For banded LU, the bound is much tighter:

\f[
\rho \leq 2^{k_l}
\f]

because only \f$k_l\f$ rows participate in each elimination step.  For typical
values \f$k_l = 1\f$ or \f$2\f$ the growth factor is bounded by 2 or 4,
making banded LU extremely stable in practice.

For **diagonally dominant** banded matrices (\f$|a_{ii}| \geq \sum_{j \neq i} |a_{ij}|\f$
for all \f$i\f$), pivoting is unnecessary: the diagonal is always the largest
element in its column, there is no fill-in beyond the original bandwidth, and the
factorisation is unconditionally stable.  Laplacian discretisations and many
radiative transfer systems fall into this category.

### Solve Phase

Given the factored \f$PA = LU\f$, solving \f$A\mathbf{x} = \mathbf{b}\f$
requires three sequential sweeps:

1. **Apply permutation**: \f$\mathbf{b} \leftarrow P\mathbf{b}\f$ using the
   stored pivot indices.
2. **Forward substitution**: solve \f$L\mathbf{y} = P\mathbf{b}\f$, touching
   at most \f$k_l\f$ off-diagonal entries per row.
3. **Back substitution**: solve \f$U\mathbf{x} = \mathbf{y}\f$, touching at
   most \f$k_l + k_u\f$ super-diagonal entries per row.

Total solve cost: \f$O(n(k_l + k_u))\f$.  For a pentadiagonal system with
\f$n = 10\,000\f$ this is roughly 40 000 FLOPs -- negligible compared to any
meaningful problem setup cost.

API: @ref num::BandedMatrix, @ref num::banded_lu, @ref num::banded_solve,
@ref num::banded_matvec
