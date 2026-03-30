# High-Performance Computing {#page_performance}

This page covers the theoretical limits that govern single-node performance, then
traces the successive optimizations applied to dense matrix multiplication in this
library: cache blocking, register blocking, and explicit SIMD vectorization.  Each
section connects the underlying hardware model to the implementation.

---

## Performance Limits {#sec_roofline}

### Why Clock Frequency Plateaued

CPU clock speeds reached approximately 3-4 GHz around 2005 and have not risen
meaningfully since.  The root cause is power dissipation.  The dynamic power
consumed by a CMOS circuit scales as

\f[
P \propto C V^2 f
\f]

where \f$C\f$ is the switching capacitance of the circuit, \f$V\f$ is the supply
voltage, and \f$f\f$ is the clock frequency.  Higher frequency requires higher
voltage to maintain signal integrity, so power scales roughly as \f$f^3\f$.
Doubling frequency would require roughly eight times the cooling capacity -- a
physical wall.

The industry response was to add more cores rather than faster cores.  Exploiting
those cores requires parallelism, which brings its own limits.

### Amdahl's Law

If a fraction \f$s\f$ of a program is inherently serial -- cannot be parallelized
regardless of hardware -- then with \f$p\f$ processors the achievable speedup is

\f[
S(p) = \frac{1}{s + \dfrac{1-s}{p}}
\f]

As \f$p \to \infty\f$ the parallel fraction vanishes and the speedup saturates at

\f[
S_{\max} = \frac{1}{s}
\f]

A program that is 5 % serial (\f$s = 0.05\f$) can never exceed 20x speedup,
regardless of how many processors are used.  Identifying and eliminating serial
bottlenecks is therefore as important as adding parallelism.

### Roofline Model

A kernel's attainable performance is bounded by whichever limit it hits first:
peak arithmetic throughput or memory bandwidth.

\f[
\text{Perf} \leq \min\!\bigl(\text{Peak FLOP/s},\; I \times \text{BW}\bigr)
\f]

\f$I\f$ is the **arithmetic intensity** -- the ratio of floating-point operations
performed to bytes transferred from main memory:

\f[
I = \frac{\text{FLOPs}}{\text{Bytes transferred}}
\f]

Kernels below the "ridge point" (where the two bounds intersect) are
**memory-bandwidth bound**; those above it are **compute bound**.

### Arithmetic Intensity of Common Kernels

| Kernel | FLOPs | Bytes read | \f$I\f$ (FLOP/byte) | Bound |
|--------|-------|------------|----------------------|-------|
| AXPY \f$\mathbf{y} \leftarrow \alpha\mathbf{x} + \mathbf{y}\f$ | \f$2n\f$ | \f$3n \times 8\f$ | \f$\tfrac{1}{12}\f$ | Memory |
| Dot product \f$\mathbf{x}^T\mathbf{y}\f$ | \f$2n\f$ | \f$2n \times 8\f$ | \f$\tfrac{1}{8}\f$ | Memory |
| Matrix-vector \f$A\mathbf{x}\f$ | \f$2n^2\f$ | \f$(n^2+n) \times 8\f$ | \f$\approx \tfrac{1}{4}\f$ | Memory |
| Matrix-matrix \f$AB\f$ | \f$2n^3\f$ | \f$3n^2 \times 8\f$ | \f$\tfrac{n}{12}\f$ | Compute (large \f$n\f$) |

Matrix-matrix multiplication is the only common dense linear algebra operation
that becomes compute-bound as \f$n\f$ grows.  All vector operations and matvec
are permanently memory-bound at practical sizes.

### Memory Hierarchy Latencies

Data flows from registers through several cache levels to DRAM.  Each level
trades capacity for speed:

| Level | Typical size | Latency | Bandwidth |
|-------|-------------|---------|-----------|
| Registers | ~256 B | 0 cycles | -- |
| L1 cache | 32-64 KB | 4 cycles | very high |
| L2 cache | 256 KB-4 MB | 12 cycles | high |
| L3 cache | 8-32 MB | 40 cycles | moderate |
| DRAM | 8-64 GB | 200+ cycles | ~50 GB/s |

A single L3 cache miss costs ~200 cycles -- enough time to execute roughly 200
floating-point multiply-adds.  Reducing cache misses is therefore frequently more
impactful than reducing the FLOP count.

Data moves between levels in 64-byte **cache lines** (8 doubles).  Loading one
double from DRAM pays for a full 64-byte transfer; sequential access amortises
that cost across 8 elements, while random access wastes 7 of them.

---

## Cache-Blocked Matrix Multiplication {#sec_cache_blocking}

### Why Naive \f$ijk\f$ Matmul is Slow

The naive triple loop over \f$i \to j \to k\f$ has the inner access pattern:

| Access | Stride as \f$k\f$ increases | Cache behaviour |
|--------|--------------------------|-----------------|
| \f$A(i,k)\f$ | +8 bytes (sequential) | stays in L1 |
| \f$B(k,j)\f$ | \f$+N \times 8\f$ bytes (column jump) | one new cache line per step |
| \f$C(i,j)\f$ | 0 (scalar accumulator) | register |

For \f$N = 512\f$, each step of the inner \f$k\f$-loop jumps 4 KB in B -- far
beyond the L1 line size.  The entire column of B is re-fetched from DRAM for
every output element \f$C(i,j)\f$.  **B is read from DRAM \f$O(N)\f$ times
instead of once.**

### Worked Cache-Miss Count for N = 512

For each of the \f$N^2\f$ output elements the inner loop touches \f$N\f$ distinct
cache lines of B (one per row, all on different lines due to the column stride).
Total B cache-line loads in the naive loop:

\f[
\frac{N^2 \times N}{8} = \frac{N^3}{8} \approx 16.8 \times 10^6 \text{ lines}
\quad (N = 512)
\f]

That is roughly 1 GB transferred from DRAM for a matrix that fits in 2 MB.

Cache blocking reduces this to

\f[
\frac{N^3}{8 \, B_s}
\f]

where \f$B_s\f$ is the tile side length.  For \f$B_s = 64\f$ and \f$N = 512\f$
the count drops to approximately 262 000 lines -- a **64x reduction**.

### The Tiling Idea

Partition A, B, and C into \f$B_s \times B_s\f$ sub-matrices and compute one
tile of C at a time.  The working set for three tiles is

\f[
3 \, B_s^2 \times 8 \text{ bytes}
\f]

Select \f$B_s\f$ so this fits in the target cache level:

| Cache level | Typical size | Working set formula | Recommended \f$B_s\f$ |
|-------------|-------------|--------------------|-----------------------|
| L1 (32 KB) | 32 768 B | \f$3 B_s^2 \times 8 \leq 32768\f$ | 32 |
| L1 (64 KB) | 65 536 B | -- | 48 or 64 |
| L2 (256 KB) | 262 144 B | -- | 64 or 96 |
| L2 (512 KB) | 524 288 B | -- | 128 |

Round \f$B_s\f$ to a multiple of the SIMD vector width (4 doubles for AVX-256,
8 for AVX-512) to assist the auto-vectoriser.  **\f$B_s = 64\f$ is the safe
default** for server CPUs with at least 256 KB L2.

### Outer Loop Order: \f$ii \to jj \to kk\f$

The outer three loops iterate over tile indices.  The ordering \f$ii \to jj \to
kk\f$ keeps the C tile \f$C[ii{:}ii{+}B_s,\, jj{:}jj{+}B_s]\f$ resident in L2
across all \f$kk\f$ iterations for a fixed \f$(ii, jj)\f$ pair.  The tile is
loaded once and written back once.  Swapping to \f$ii \to kk \to jj\f$ would
evict the C tile between \f$jj\f$ iterations, sacrificing half the benefit.

### Inner Loop Order: \f$i \to k \to j\f$

Inside each tile, the loop order matters too.  The \f$i \to k \to j\f$ order
hoists the scalar \f$A(i,k)\f$ into a register before the innermost \f$j\f$-loop:

\f[
\mathbf{c}_{i,\,jj:jj+B_s} \mathrel{+}= A(i,k) \cdot \mathbf{b}_{k,\,jj:jj+B_s}
\f]

The innermost loop is a **scaled vector addition (AXPY)** over sequential memory
in both B and C -- the shape the compiler needs to auto-vectorise with SIMD.  The
alternative \f$i \to j \to k\f$ ordering reintroduces column-stride access to B
within the tile, reproducing the original problem at smaller scale.

### Measured Speedup

Benchmarks compiled with `-O2`, single-threaded, on an 8-core machine with L2
4 MB per core:

| N | Naive | Blocked (\f$B_s = 64\f$) | Speedup |
|---|-------|--------------------------|---------|
| 64 | 160 mus | 32 mus | **5.0x** |
| 128 | 1 842 mus | 263 mus | **7.0x** |
| 256 | 18 135 mus | 2 761 mus | **6.6x** |
| 512 | 158 131 mus | 25 036 mus | **6.3x** |

The BigO constant drops from **1.18 -> 0.19** -- a consistent 6x improvement with
no parallelism and no explicit SIMD.  Both variants remain \f$O(N^3)\f$; only the
constant changes because cache-miss count per FLOP falls dramatically.

API: @ref num::matmul_blocked

---

## Register Blocking {#sec_register_blocking}

### The Residual Problem After Cache Blocking

Even with the B tile resident in L2, the cache-blocked micro-kernel still
performs a load-modify-store of \f$C(i,j)\f$ on every \f$k\f$-step:

```
load C(i,j)  ->  add a_ik * B(k,j)  ->  store C(i,j)
```

For a 64x64 cache tile with \f$K = 512\f$ that is \f$4096 \times 512 = 2\text{M}\f$
L2 load/store operations.  The solution is to keep a small sub-tile of C entirely
in **registers** (local variables), accumulate across the full \f$kk\f$ block, then
write C back to L2 only once per tile.

### The Register Tile

Declare a REGxREG array of `double` locals -- for REG = 4 this fits in 16 scalar
variables, mapped to 8 YMM registers under AVX-256 (two doubles per register when
scalars are paired by the compiler).  The structure is:

1. **Load** the existing partial sum from L2 into the local \f$c[i][j]\f$ array.
2. **Accumulate** the entire \f$kk\f$ block: \f$c[i][j] \mathrel{+}= A(i,k) \cdot B(k,j)\f$ with no C memory traffic.
3. **Write back** the completed partial sum to L2 once.

The \f$kk\f$ loop remains **outside** the register tile, preserving B-tile L2
residency exactly as in cache blocking.

### Why Scalar Register Blocking Regresses

In isolation, scalar register blocking is **3x slower** than plain cache blocking
(measured BigO constant 0.57 vs 0.18).  The explanation is vectorisation:

The cache-blocked micro-kernel's innermost \f$j\f$-loop runs for 64 iterations over
sequential memory.  The compiler auto-vectorises this as 16 AVX-256 FMA
instructions.  The register-blocked \f$j\f$-loop runs for only 4 iterations -- too
short to amortise loop overhead, and too small for the compiler to reliably
vectorise.  The savings on C memory traffic are real but smaller than the added
overhead of the extra \f$ir/jr\f$ loop nesting.

### Register Blocking + SIMD as a Unit

Register blocking only pays off when the tile width equals the SIMD vector width,
so the "inner loop" collapses into a **single vector instruction**:

| Tile width | Scalar | AVX-256 | AVX-512 |
|------------|--------|---------|---------|
| Doubles per FMA | 1 | **4** | 8 |
| \f$j\f$-loop iterations | 4 | **1** | 1 |
| Loop overhead | 4 steps | **0 steps** | 0 steps |

When the 4-iteration scalar loop becomes one `vfmadd256` instruction, all overhead
disappears.  A 4x4 register tile then exposes **four independent FMA chains** to
the out-of-order scheduler, allowing the two FMA execution ports on modern x86
(Skylake and later) to stay fully occupied.

The BLAS micro-kernel from Goto (2008) is exactly this structure: four independent
YMM accumulators (c0, c1, c2, c3) updated in parallel across the \f$k\f$-loop,
written back to C once after the loop.

API: @ref num::matmul_register_blocked

---

## SIMD Vectorization {#sec_simd}

### AVX-256 and ARM NEON

Two SIMD instruction sets are relevant to this library:

| ISA | Register width | Doubles per register | Key FMA instruction |
|-----|---------------|----------------------|---------------------|
| AVX-256 (x86-64) | 256 bits | 4 | `_mm256_fmadd_pd` |
| ARM NEON (AArch64) | 128 bits | 2 | `vfmaq_f64` |

The AVX-256 fused multiply-add \f$\_mm256\_fmadd\_pd\f$ computes
\f$\mathbf{c} \mathrel{+}= \mathbf{a} \times \mathbf{b}\f$ for 4 doubles
simultaneously in a single instruction.  With two FMA execution ports running at
full speed the theoretical peak throughput is

\f[
\text{Peak GFLOP/s} = 2 \times 8 \times f_{\text{clock}}
\f]

where the factor of 2 is the FMA (one multiply + one add), 8 is the number of
doubles per cycle across both ports (4 doubles x 2 ports), and \f$f_{\text{clock}}\f$
is the sustained clock frequency in GHz.  ARM NEON's `vfmaq_f64` processes 2
doubles per instruction; on Apple M-series silicon the M core's NEON FMA
throughput is correspondingly half that of AVX-256 in terms of doubles per cycle.

### The SIMD Micro-kernel

For the 4x4 register tile the \f$k\f$-loop body in AVX-256 is:

```
b   = load  B[k, jr..jr+3]         // 1 vmovupd  -- 4 consecutive doubles
c0 += broadcast(A[ir+0, k]) * b    // 1 vbroadcastsd + 1 vfmadd
c1 += broadcast(A[ir+1, k]) * b    // independent
c2 += broadcast(A[ir+2, k]) * b    // independent
c3 += broadcast(A[ir+3, k]) * b    // independent
```

Four independent FMA chains allow the CPU's out-of-order engine to keep both FMA
ports occupied every cycle.  For NEON the same tile uses 8 `vfmaq_f64`
instructions per \f$k\f$-step (two NEON vectors cover the 4-wide column).

### Compile-Time Architecture Dispatch

The CMake build detects the host architecture at configure time and defines one of:

- `NUMERICS_HAS_AVX2` -- x86-64 with AVX2 + FMA3 (enabled via `-mavx2 -mfma`)
- `NUMERICS_HAS_NEON` -- AArch64 (NEON is mandatory, no flag needed)

The public entry point `matmul_simd` dispatches to the correct backend at compile
time:

```cpp
void matmul_simd(const Matrix& A, const Matrix& B, Matrix& C, idx block_size)
{
#if defined(NUMERICS_HAS_AVX2)
    matmul_avx(A, B, C, block_size);
#elif defined(NUMERICS_HAS_NEON)
    matmul_neon(A, B, C, block_size);
#else
    matmul_blocked(A, B, C, block_size);   // scalar fallback
#endif
}
```

The fallback guarantees correct execution on any architecture, which is important
for CI pipelines that build on heterogeneous machines.  Note that `-mfma` is
required on x86 to emit the fused instruction encoding; without it the compiler
emits separate `vmulpd` + `vaddpd` pairs -- same FLOP count, twice the instruction
count.

### Jacobi SVD SIMD Fusion

The Jacobi SVD inner loop involves repeated inner products and Givens column
rotations.  In the SIMD path these two passes are fused: inner-product partial
sums and the rotation update share the same loaded data, halving global-memory
traffic for the column pair.  The dispatch follows the same
`NUMERICS_HAS_AVX2 / NUMERICS_HAS_NEON / fallback` pattern used by matmul.

### Measured Progression

All benchmarks are single-threaded, compiled with `-O2`:

| Kernel | BigO constant | vs naive |
|--------|--------------|---------|
| Naive \f$ijk\f$ | 1.19 \f$N^3\f$ | baseline |
| Cache-blocked | 0.18 \f$N^3\f$ | **6.6x** |
| SIMD (NEON) | 0.09 \f$N^3\f$ | **13.9x** |
| OpenBLAS equivalent | ~0.005 \f$N^3\f$ | remaining gap ~18x |

At \f$N = 512\f$ the SIMD kernel runs in 11 498 mus versus 159 553 mus for naive --
a wall-time reduction of 13.9x.  The remaining gap to OpenBLAS is closed by wider
tiles, B-panel packing (eliminating residual stride penalties), software prefetch,
and multithreading.

### matvec\_simd: Bandwidth Bound

Matrix-vector multiply is memory-bandwidth bound: every element of A is read
exactly once.  The SIMD path gives a modest 1.5-1.8x improvement because the
bottleneck shifts from instruction throughput to memory bandwidth -- no amount of
vectorisation can exceed the available bandwidth.

| N | Scalar matvec | SIMD matvec | Speedup |
|---|--------------|-------------|---------|
| 64 | 1 469 ns | 829 ns | 1.8x |
| 512 | 191 685 ns | 114 969 ns | 1.7x |
| 2048 | 3 744 813 ns | 2 445 977 ns | 1.5x |

API: @ref num::matmul_simd, @ref num::matvec_simd
