# Week 6: Cache-Aware Computing -- Blocked Matrix Multiplication {#page_week6}

## Overview

This week we move from *algorithmic* complexity to *machine* complexity.  Two
programs can perform the same number of floating-point operations and yet run
10x apart in wall time, because one of them respects the cache hierarchy and
the other does not.

Dense matrix multiplication is the canonical example.  Its naive implementation
is O(N^3) in FLOPs *and* in cache misses -- and the cache misses dominate.  A
cache-blocked version achieves the same asymptotic complexity but cuts the
constant by **6x without any parallelism or SIMD**, purely through data
reuse.

---

## 1. The Memory Hierarchy

Every modern CPU has a multi-level cache between the processor registers and
main memory.  Each level trades capacity for latency:

```
         Registers   ~  256 B   |  0 cycles
         -------------------------------------
         L1 cache    ~ 32-64 KB |  4 cycles
         L2 cache    ~ 256 KB-4 MB | 12 cycles
         L3 cache    ~ 8-32 MB  |  40 cycles
         -------------------------------------
         DRAM        ~ 8-64 GB  | 200+ cycles
```

A **cache miss** occurs when data is not in cache and must be fetched from a
slower level.  A single L3 miss costs ~200 cycles -- enough time to execute
~200 floating-point multiply-adds.  Eliminating unnecessary misses is often
more impactful than reducing FLOPs.

Data moves between levels in fixed-size chunks called **cache lines** (typically
64 bytes = 8 doubles).  When you load one double from DRAM, you pay for a full
64-byte transfer.  Sequential access patterns amortise that cost across 8
elements; random access wastes 7 of them.

---

## 2. Arithmetic Intensity and the Roofline Model

**Arithmetic intensity** (AI) measures how much computation you extract per byte
moved from memory:

\f[\text{AI} = \frac{\text{FLOPs}}{\text{bytes transferred}}\f]

The **roofline model** says a kernel's performance is bounded by whichever
limit it hits first:

\f[\text{Performance} \leq \min\!\left(\text{Peak FLOP/s},\ \text{AI} \times \text{Memory BW}\right)\f]

For naive NxN matrix multiply:

| Quantity | Value |
|---|---|
| FLOPs | \f$2N^3\f$ (N^3 multiplies + N^3 adds) |
| Bytes read (naive) | \f$\approx 2N^3 \times 8\f$ (B read N^2 times, once per column element per row) |
| Arithmetic intensity | \f$\approx 0.125\f$ FLOP/byte |

0.125 FLOP/byte is **deep in the memory-bandwidth-bound** regime on every
known CPU.  A typical server has 50 GB/s of memory bandwidth and 1 TFLOP/s of
peak compute -- the ridge point (where compute and memory limits meet) is at
50/1000 x 1e9 = ~20 FLOP/byte.  We are 160x below it.

Blocking raises the arithmetic intensity toward that ridge point by reusing
data already in cache rather than re-fetching it from DRAM.

---

## 3. Why Naive Matrix Multiply is Slow

### Code (`src/core/matrix.cpp:87-97`)

```cpp
// Naive i-j-k baseline
for (idx i = 0; i < A.rows(); ++i) {
    for (idx j = 0; j < B.cols(); ++j) {
        C(i, j) = 0;
        for (idx k = 0; k < A.cols(); ++k)
            C(i, j) += A(i, k) * B(k, j);   // <- B access is the problem
    }
}
```

All three matrices are stored **row-major** (C-style): element `M(i, j)` sits
at address `base + (i * cols + j) * sizeof(double)`.

Trace the inner k-loop for a fixed output element `C(i, j)`:

| Access | Address stride as k increases | Cache behaviour |
|---|---|---|
| `A(i, k)` | +8 bytes (sequential) | ok prefetchable, stays in L1 |
| `B(k, j)` | +`N x 8` bytes (column jump) | x one new cache line per step |
| `C(i, j)` | 0 (scalar accumulator) | ok register |

`B(k, j)` for fixed `j` and varying `k` is a **column** of B -- elements spaced
`N` doubles apart.  For N = 512, that stride is 4 KB per step.  A 32 KB L1
cache holds 512 doubles.  After 512 steps the entire L1 has been cycled through
for column `j` -- and then the *next* `j` starts the same eviction pattern
from scratch.  **B is read from DRAM O(N) times instead of once.**

For N = 512 the naive loop reads B ~= 512 x (512 x 512 x 8) / 10^9 ~= 1 GB from
DRAM.  With 50 GB/s bandwidth that is 20 ms of stalls -- on top of the actual
arithmetic.

---

## 4. The Blocking Idea

Instead of iterating over entire rows and columns, **tile** A, B and C into
BLOCKxBLOCK sub-matrices and compute one tile at a time:

```
  A                    B                    C
+---+---+---+      +---+---+---+      +---+---+---+
|   |   |   |      |B00|B01|B02|      |   |   |   |
+---+---+---+  x   +---+---+---+  =   +---+---+---+
|A10|A11|A12|      |   |   |   |      |   |C11|   |
+---+---+---+      +---+---+---+      +---+---+---+
|   |   |   |      |   |   |   |      |   |   |   |
+---+---+---+      +---+---+---+      +---+---+---+

  C(1,1) += A(1,0)xB(0,1) + A(1,1)xB(1,1) + A(1,2)xB(2,1)
              one tile of A   one tile of B     accumulated into C tile
```

The working set for three BLOCKxBLOCK tiles of doubles is:

\f[\text{working set} = 3 \times \text{BLOCK}^2 \times 8\ \text{bytes}\f]

| BLOCK | Working set | Fits in |
|---|---|---|
| 32 | 24 KB | L1 (32 KB) |
| 64 | 98 KB | L2 (256 KB) |
| 128 | 393 KB | L2 (512 KB) or L3 |

Pick BLOCK so the working set fits in the cache level you want to exploit.
**BLOCK = 64** is a robust default for server CPUs with 256 KB L2.

Within each tile, B's elements are accessed left-to-right along rows of a
BLOCKxBLOCK sub-matrix -- sequential, fully prefetchable, and staying in L2 for
the duration of the tile computation.  The column-stride problem disappears.

---

## 5. The Algorithm

### Outer loop structure (`src/core/matrix.cpp:153-161`)

```
for ii in 0, BLOCK, 2*BLOCK, ... M:          <- tile row index of A and C
  for jj in 0, BLOCK, 2*BLOCK, ... N:        <- tile column index of B and C
    for kk in 0, BLOCK, 2*BLOCK, ... K:      <- tile index along shared dimension
      micro-kernel(A[ii:ii+B, kk:kk+B],
                   B[kk:kk+B, jj:jj+B],
                   C[ii:ii+B, jj:jj+B])
```

**Why `ii -> jj -> kk` and not some other order?**

For fixed `(ii, jj)`, the C tile `C[ii:i_end, jj:j_end]` is updated by every
`kk` block.  With the `ii-jj-kk` ordering it stays in L2 across *all* `kk`
iterations -- loaded once, updated BLOCK times, written back once.  If we
swapped to `ii-kk-jj` the C tile would be evicted between `jj` iterations and
reloaded from L2/L3 each time, losing half the benefit.

### Micro-kernel (`src/core/matrix.cpp:163-170`)

```cpp
for (idx i = ii; i < i_end; ++i) {
    for (idx k = kk; k < k_end; ++k) {
        const real a_ik = A(i, k);          // <- scalar: hoisted into register
        for (idx j = jj; j < j_end; ++j)
            C(i, j) += a_ik * B(k, j);     // <- sequential B and C
    }
}
```

**Why `i -> k -> j` and not `i -> j -> k`?**

Compare the two orderings inside the tile:

```
i-j-k (bad):                    i-k-j (good):
  for i:                          for i:
    for j:                          for k:
      for k:                          a_ik = A(i,k)   <- register
        C(i,j) += A(i,k)*B(k,j)       for j:
                                         C(i,j) += a_ik * B(k,j)
```

In `i-j-k`, `B(k,j)` for fixed `j` and varying `k` is again a column stride.
We have replicated the original problem *within* the tile.

In `i-k-j`, `A(i,k)` is a scalar that the compiler hoists before the `j` loop.
The innermost loop becomes a **scaled vector addition** (AXPY):

\f[\mathbf{c}_{i,jj:j\_end} \mathrel{+}= a_{ik} \cdot \mathbf{b}_{k,jj:j\_end}\f]

Both `B(k,j)` and `C(i,j)` are read/written left-to-right along rows --
sequential memory access, unit stride.  The compiler can auto-vectorise this
loop with SIMD instructions (AVX-256 processes 4 doubles at once; AVX-512
processes 8).

---

## 6. Implementation Walkthrough

### Declaration (`include/core/matrix.hpp:48-64`)

```cpp
void matmul(const Matrix& A, const Matrix& B, Matrix& C,
            Exec exec = Exec::cpu);          // naive baseline

void matmul_blocked(const Matrix& A, const Matrix& B, Matrix& C,
                    idx block_size = 64);    // cache-blocked
```

Both functions are kept side-by-side so you can benchmark them directly.
`block_size` is a runtime parameter so experiments at different tile sizes do
not require recompilation.

### Full implementation (`src/core/matrix.cpp:148-172`)

```cpp
void matmul_blocked(const Matrix& A, const Matrix& B, Matrix& C, idx block_size) {
    const idx M = A.rows(), K = A.cols(), N = B.cols();

    std::fill_n(C.data(), M * N, real(0));      // zero C before accumulating

    for (idx ii = 0; ii < M; ii += block_size) {
        const idx i_end = std::min(ii + block_size, M);    // boundary guard

        for (idx jj = 0; jj < N; jj += block_size) {
            const idx j_end = std::min(jj + block_size, N);

            // -- C tile stays in L2 across all kk iterations --------------
            for (idx kk = 0; kk < K; kk += block_size) {
                const idx k_end = std::min(kk + block_size, K);

                for (idx i = ii; i < i_end; ++i) {
                    for (idx k = kk; k < k_end; ++k) {
                        const real a_ik = A(i, k);       // register scalar
                        for (idx j = jj; j < j_end; ++j) // auto-vectorised
                            C(i, j) += a_ik * B(k, j);
                    }
                }
            }
        }
    }
}
```

The boundary guards (`std::min`) handle matrices whose dimensions are not exact
multiples of `block_size` -- no padding required.

---

## 7. Benchmark Results

Measured on an 8-core machine (L1d 64 KB, L2 4 MB per core).
Compiled with `-O2`.  Single-threaded, CPU only.

```
Benchmark                         Time        CPU     Complexity
-----------------------------------------------------------------
BM_Matmul_CPU/64             160351 ns   160241 ns
BM_Matmul_CPU/128           1842283 ns  1841221 ns
BM_Matmul_CPU/256          18134806 ns 18125462 ns
BM_Matmul_CPU/512         158131521 ns 158088750 ns
BM_Matmul_CPU_BigO              1.18 N^3

BM_MatmulBlocked_CPU/64       32382 ns    32360 ns
BM_MatmulBlocked_CPU/128     262533 ns   262432 ns
BM_MatmulBlocked_CPU/256    2760950 ns  2759480 ns
BM_MatmulBlocked_CPU/512   25036320 ns 25012143 ns
BM_MatmulBlocked_CPU_BigO       0.19 N^3
```

| N | Naive | Blocked | Speedup |
|---|---|---|---|
| 64 | 160 mus | 32 mus | **5.0x** |
| 128 | 1842 mus | 263 mus | **7.0x** |
| 256 | 18135 mus | 2761 mus | **6.6x** |
| 512 | 158131 mus | 25036 mus | **6.3x** |

The BigO coefficient drops from **1.18 -> 0.19** -- a consistent **~6.3x**
improvement.  Both are still O(N^3); the constant changes because the number of
cache misses per FLOP falls dramatically.

Reproduce with:
```bash
cmake --build build --target numerics_bench
./build/benchmarks/numerics_bench --benchmark_filter="BM_Matmul"
```

Benchmark source: `benchmarks/bench_linalg.cpp:93-135`

---

## 8. Worked Cache-Miss Count

Let N = 512, BLOCK = 64, cache line = 64 bytes (8 doubles).

### Naive loop -- B accesses

For each of N^2 output elements `C(i,j)`, the inner k-loop reads the entire
column `B(:, j)` -- N elements, all on different cache lines (stride 4 KB >> 64
bytes).  Total B cache lines loaded:

\f[\frac{N^2 \times N}{8} = \frac{512^3}{8} \approx 16.8 \times 10^6\ \text{lines}\f]

Each cache line is 64 bytes -> **1.07 GB** of B transferred from DRAM
(for N=512, the matrix is only 2 MB).  B is re-read ~500 times.

### Blocked loop -- B accesses

Each B tile of size BLOCKxBLOCK is loaded once per `(ii, kk)` pair and reused
across all `jj` iterations for that `(ii, kk)`.  Wait -- actually in the
`ii-jj-kk` ordering a B tile is loaded once per `(ii, jj)` pair... let me be
precise.

For fixed `(ii, jj, kk)`, the B tile `B[kk:kk+B, jj:jj+B]` is loaded once and
used for all `i` in `[ii, ii+B)`.  It is loaded again the next time that
`(jj, kk)` combination appears, which is for the next `ii` block.  So each B
tile is loaded `M/BLOCK` times total.

Total B cache lines loaded:

\f[\frac{(N/\text{BLOCK})^3 \times \text{BLOCK}^2}{8} = \frac{N^3}{8 \times \text{BLOCK}}\f]

For N=512, BLOCK=64:

\f[\frac{512^3}{8 \times 64} = \frac{16.8 \times 10^6}{64} \approx 262\,000\ \text{lines}\f]

That is **64x fewer cache lines** -- matching the speedup order of magnitude.
(The measured 6x rather than 64x reflects that the naive loop benefits from
L2/L3 caching for small N, and the blocked loop has its own overhead for
boundary handling and loop control.)

---

## 9. Choosing the Block Size

The goal is to fit three tiles in your target cache level.

```
BLOCK = floor( sqrt( cache_size_bytes / (3 x 8) ) )
```

| Cache level | Typical size | Optimal BLOCK |
|---|---|---|
| L1 (32 KB) | 32,768 bytes | 37 -> use **32** |
| L1 (64 KB) | 65,536 bytes | 52 -> use **48** or **64** |
| L2 (256 KB) | 262,144 bytes | 104 -> use **64** or **96** |
| L2 (512 KB) | 524,288 bytes | 148 -> use **128** |

Round down to a power of 2 or multiple of the SIMD width (4 doubles for
AVX-256, 8 for AVX-512) to help the vectoriser.

In practice, **BLOCK = 64 is the safe default** for modern server CPUs with at
least 256 KB L2.  For a tuning exercise, benchmark `matmul_blocked` at
BLOCK = 32, 48, 64, 96, 128 on your specific machine and pick the knee of the
curve.

---

## 10. What Comes Next

This week's implementation leaves significant performance on the table.  The
0.19 N^3 constant compares to OpenBLAS's ~0.005 N^3 -- roughly **40x gap**
remaining.  That gap is closed in two further steps:

### Step 1 -- Register blocking

Instead of accumulating into `C(i,j)` in memory, accumulate into a small
register tile (e.g. 4x4 or 8x4 doubles) declared as local variables.  This
eliminates the `C` load/store in the inner loop entirely and raises register
reuse.  The inner loop transforms from:

```cpp
C(i, j) += a_ik * B(k, j);      // <- load C, add, store C every iteration
```

to accumulating into `double c00, c01, c02, c03, ...` and writing back to C only
after the micro-kernel completes.

### Step 2 -- Explicit SIMD

The compiler *sometimes* auto-vectorises the `j` loop, but it cannot always
prove safety (aliasing, alignment).  Explicit SIMD intrinsics (AVX-256 on
x86-64, NEON on ARMv8) guarantee vectorisation and unlock fused
multiply-add (FMA) instructions:

```cpp
// AVX-256 conceptually:
__m256d a_vec = _mm256_set1_pd(a_ik);        // broadcast scalar to 4 lanes
__m256d b_vec = _mm256_loadu_pd(&B(k, jj));  // load 4 doubles
__m256d c_vec = _mm256_loadu_pd(&C(i, jj));
c_vec = _mm256_fmadd_pd(a_vec, b_vec, c_vec); // c += a * b  (single instruction)
_mm256_storeu_pd(&C(i, jj), c_vec);
```

Each FMA processes 4 doubles simultaneously.  With 8 AVX-512 FMAs per cycle
and two ports, peak throughput on a modern CPU is 2 x 8 = 16 doubles/cycle.

---

## 11. Key Takeaways

1. **Cache misses dominate** in memory-bound kernels.  For naive matmul, B is
   read O(N) times from DRAM; blocking reduces this to O(1) per tile.

2. **Arithmetic intensity** is the design target.  Cache blocking raises AI
   from ~0.125 FLOP/byte toward the ridge point of the roofline model.

3. **Loop order matters** inside the tile.  The `i -> k -> j` inner order hoists
   a scalar into a register and leaves the innermost loop as a sequential AXPY
   -- the shape the compiler needs to auto-vectorise.

4. **Block size is a tuning parameter**, not a magic constant.  Derive it from
   your cache size; round to a SIMD-friendly value.

5. **6x from reordering alone, no parallelism, no SIMD.**  This is the first
   step in a hierarchy: blocking -> register tiling -> SIMD -> threading.  Each
   step compounds the previous one.

---

## Exercises

1. Run `BM_MatmulBlocked_CPU` with `block_size` set to 32, 48, 64, 96, 128
   and plot wall time vs block size.  Identify the optimal value for your
   machine.  Does it match the formula in Sec.9?

2. Add a `matmul_blocked` call to the CG solver benchmark and measure whether
   CG benefits (it uses `matvec`, not `matmul` -- what does that tell you about
   where to apply blocking next?).

3. Instrument the naive and blocked loops with `perf stat -e cache-misses` on
   Linux.  Verify that the cache-miss count drops by roughly the factor
   predicted in Sec.8.

4. Implement a version that uses two levels of blocking: an outer tile for L3
   and an inner tile for L2.  What is the working set formula for each level?

5. *(Advanced)* Add `#pragma omp parallel for` to the outer `ii` loop of
   `matmul_blocked`.  What speedup do you observe on 4 threads?  Is it linear?
   What limits it?

---

## References

- Goto, K. & van de Geijn, R. (2008). *Anatomy of high-performance matrix
  multiplication.* ACM TOMS 34(3). -- The paper behind every modern BLAS dgemm.
- Lam, M., Rothberg, E., & Wolf, M. (1991). *The cache performance and
  optimizations of blocked algorithms.* ASPLOS IV. -- Original tiling paper.
- Williams, S., Waterman, A., & Patterson, D. (2009). *Roofline: An insightful
  visual performance model.* CACM 52(4).
