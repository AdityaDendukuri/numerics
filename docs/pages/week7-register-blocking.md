# Week 7: Register Blocking -- What It Is and Why It Needs SIMD {#page_week7}

## Overview

Last week's cache-blocked `matmul_blocked` cut wall time by **6x** by keeping
the B tile resident in L2 cache.  This week we look at the next optimization
in the BLAS hierarchy -- register blocking -- implement it, measure it, and
discover that it does **not** help in isolation.  Understanding *why* reveals
the tight coupling between register blocking and explicit SIMD, and sets up
week 8 directly.

---

## 1. The Residual Bottleneck After Cache Blocking

Look at the micro-kernel from last week
(`src/core/matrix.cpp:165-170`):

```cpp
for (idx i = ii; i < i_end; ++i) {
    for (idx k = kk; k < k_end; ++k) {
        const real a_ik = A(i, k);
        for (idx j = jj; j < j_end; ++j)
            C(i, j) += a_ik * B(k, j);   // <- C is a memory address
    }
}
```

`C(i, j)` lives in L2 cache (that's the point of cache blocking), but every
k-step does a read-modify-write:

```
load  C(i,j)   ->  add a_ik * B(k,j)  ->  store C(i,j)
```

For a 64x64 cache tile with K=512, that is 4096 x 512 = **2M load/store
cycles** against L2.  In theory, accumulating into registers instead and
writing C back once per kk block should cut this 64-fold.

---

## 2. Register Blocking: The Idea

Subdivide the cache tile into REGxREG "register tiles".  Declare the C
sub-block as local `double` variables, accumulate for the entire kk block in
registers, then write back to L2 once:

```cpp
real c[4][4] = {};                      // lives in registers (if compiler cooperates)

// load existing C partial sum (from previous kk blocks)
for i, j in tile: c[i][j] = C(i, j);

// accumulate kk block -- no C memory traffic
for k in [kk, kk+B):
    for i in tile:
        a_ik = A(i, k)
        for j in tile:
            c[i][j] += a_ik * B(k, j)   // pure register FMA

// write back once
for i, j in tile: C(i, j) = c[i][j];
```

The kk loop stays **outside** the register tile loop (preserving B-tile L2
residency), so the B tile is loaded once per `(ii, jj, kk)` triplet, exactly
as in `matmul_blocked`.

---

## 3. Implementation

### Declaration (`include/core/matrix.hpp:63-77`)

```cpp
void matmul_register_blocked(const Matrix& A, const Matrix& B, Matrix& C,
                              idx block_size = 64, idx reg_size = 4);
```

### Full implementation (`src/core/matrix.cpp`)

```cpp
void matmul_register_blocked(const Matrix& A, const Matrix& B, Matrix& C,
                              idx block_size, idx reg_size) {
    const idx M = A.rows(), K = A.cols(), N = B.cols();
    std::fill_n(C.data(), M * N, real(0));

    for (idx ii = 0; ii < M; ii += block_size) {
        const idx i_lim = std::min(ii + block_size, M);
        for (idx jj = 0; jj < N; jj += block_size) {
            const idx j_lim = std::min(jj + block_size, N);

            // kk is OUTSIDE the register tile -- B tile stays in L2.
            for (idx kk = 0; kk < K; kk += block_size) {
                const idx k_lim = std::min(kk + block_size, K);

                for (idx ir = ii; ir < i_lim; ir += reg_size) {
                    const idx ri = std::min(ir + reg_size, i_lim);
                    for (idx jr = jj; jr < j_lim; jr += reg_size) {
                        const idx rj = std::min(jr + reg_size, j_lim);

                        real c[4][4] = {};
                        for (idx i = ir; i < ri; ++i)
                            for (idx j = jr; j < rj; ++j)
                                c[i - ir][j - jr] = C(i, j);   // load from L2

                        for (idx k = kk; k < k_lim; ++k)
                            for (idx i = ir; i < ri; ++i) {
                                const real a_ik = A(i, k);
                                for (idx j = jr; j < rj; ++j)
                                    c[i - ir][j - jr] += a_ik * B(k, j);
                            }

                        for (idx i = ir; i < ri; ++i)
                            for (idx j = jr; j < rj; ++j)
                                C(i, j) = c[i - ir][j - jr];  // write to L2
                    }
                }
            }
        }
    }
}
```

---

## 4. Benchmark Results -- A Surprise

```bash
cmake --build build --target numerics_bench
./build/benchmarks/numerics_bench --benchmark_filter="BM_Matmul"
```

```
Benchmark                            Time        CPU     Complexity
--------------------------------------------------------------------
BM_Matmul_CPU/64               159786 ns  159728 ns
BM_Matmul_CPU/512            158332583 ns
BM_Matmul_CPU_BigO                1.18 N^3

BM_MatmulBlocked_CPU/64          32299 ns   32289 ns
BM_MatmulBlocked_CPU/512      24762444 ns
BM_MatmulBlocked_CPU_BigO         0.18 N^3

BM_MatmulRegBlocked_CPU/64      138067 ns  138018 ns    <- slower!
BM_MatmulRegBlocked_CPU/512   77196412 ns
BM_MatmulRegBlocked_CPU_BigO       0.57 N^3              <- slower!
```

Register blocking is **3x slower** than plain cache blocking.  The theory
promised a 64x reduction in C traffic.  What went wrong?

---

## 5. Why It Didn't Help: The Vectorisation Problem

### What matmul_blocked does well

The inner j-loop in `matmul_blocked` runs for **64 iterations** (the full
cache tile width):

```cpp
for (idx j = jj; j < j_end; ++j)   // 64 iterations
    C(i, j) += a_ik * B(k, j);
```

With 64 sequential doubles, the compiler auto-vectorises this as 16 AVX-256
FMA instructions (4 doubles per instruction).  Peak throughput on one core.

### What register blocking does

The inner j-loop in the register tile runs for **4 iterations**:

```cpp
for (idx j = jr; j < rj; ++j)      // 4 iterations
    c[i-ir][j-jr] += a_ik * B(k, j);
```

4 iterations = 1 AVX-256 vector.  There is barely any loop to vectorise.  The
compiler may emit one `vfmadd` and call it done -- which is the same throughput
as before, but now wrapped in 256x more loop overhead (the `ir` and `jr`
loops, plus the load/write-back passes over `c`).

**The savings on C traffic are real but smaller than the added overhead.**

More precisely: saving one L2 read+write per element per kk block gains ~24
cycles.  But the `ir/jr` loop structure adds ~30 cycles of overhead per
register tile iteration.  Net: negative.

### The hardware angle

Modern out-of-order CPUs (especially Apple M-series with massive OOO windows
and fast store-to-load forwarding) can already overlap the C load/store with
the preceding/following FMA.  The hardware is doing manually what register
blocking tries to do in software -- so the software version loses the overhead
game.

---

## 6. The Correct Mental Model: Register Blocking + SIMD Together

Register blocking is not a standalone optimization.  In every production BLAS
(OpenBLAS, MKL, BLIS), the register tile is the **SIMD unit**, not a scalar
loop:

| Tile width | Scalar | AVX-256 | AVX-512 |
|---|---|---|---|
| Doubles per FMA | 1 | **4** | 8 |
| Register tile j-width | 1-4 | **4** | 8 |
| j-loop iterations | 4 | **1** | 1 |

When `j = jr` to `jr + 4` maps to a **single `vfmadd` instruction** (not a
4-iteration scalar loop), all of the register tile overhead disappears.  The
4x4 tile becomes 4 rows x 1 SIMD operation = 4 independent `vfmadd` chains
running in parallel on the CPU's two FMA ports.

The register tile is not there to replace a scalar loop with a slightly tighter
scalar loop.  It is the structure that **exposes independent FMA chains to the
instruction scheduler**, so the CPU can keep both FMA ports busy at all times.

---

## 7. The BLAS Micro-kernel Design

The full BLAS dgemm micro-kernel (from the Goto 2008 paper) looks like this:

```
Panel layout (B panel): pack B[kk:kk+B, jj:jj+B] into a contiguous buffer
                         -> eliminates stride on B reads, enables prefetch

Register tile (4 rows x 4 cols with AVX-256):
    YMM accumulators:  c0, c1, c2, c3       <- 4 YMM registers (one per row)
    for k = 0 .. B-1:
        broadcast A(i, k) -> YMM scalar
        YMM b = load  Bpanel[k, jj..jj+3]  <- sequential, 1 vector load
        vfmadd c0 += a0 * b                 <- row 0
        vfmadd c1 += a1 * b                 <- row 1  (independent)
        vfmadd c2 += a2 * b                 <- row 2  (independent)
        vfmadd c3 += a3 * b                 <- row 3  (independent)
    store c0..c3 -> C
```

Four independent FMA chains means 4x instruction-level parallelism.  With two
FMA ports issuing per cycle, peak throughput is 2 x 4 doubles = 8 doubles per
cycle (or 8 x 4 = 32 with AVX-256 counting all elements).

Our scalar `c[4][4]` implementation was a prototype of this design -- correct
in structure, but missing the SIMD width that makes the structure pay off.

---

## 8. What We Learned

| Optimization | Result | Why |
|---|---|---|
| Cache blocking | **6x faster** | Eliminates DRAM misses for B |
| Register blocking (scalar) | **3x slower** | Breaks j-loop vectorization, adds overhead |
| Register blocking + SIMD | Expected **4-8x on top of cache** | j-loop -> 1 `vfmadd`, 4 FMA chains |

Register blocking is not wrong -- it is necessary.  But it only pays off when
the register tile width equals the SIMD vector width, so the "inner loop"
disappears entirely into a single vector instruction.

This is why OpenBLAS ships hand-written assembly micro-kernels for each
architecture: the SIMD intrinsics cannot be abstracted away without losing
the performance that makes the register tile worthwhile.

---

## 9. Progression and Next Step

```
Naive i-j-k           1.18 N^3  (baseline)
+ cache blocking       0.18 N^3  (6.5x gain)
+ register (scalar)    0.57 N^3  (regression -- wrong tool in isolation)
+ explicit SIMD        ~0.05 N^3  (projected, next week)
----------------------------------------------------------------------
OpenBLAS               ~0.005 N^3
```

Next week: replace the scalar j-loop with AVX-256 intrinsics
(`_mm256_fmadd_pd`), making each j-step process 4 doubles simultaneously.
At that point the 4x4 register tile earns its keep.

---

## 10. Key Takeaways

1. **Register blocking is real, but it needs SIMD to pay off.**  A scalar
   4-iteration j-loop has the same throughput as a vectorised 64-iteration
   loop -- but with far more loop overhead.

2. **Shrinking the inner loop hurts auto-vectorisation.**  The compiler
   produces its best code on long, simple, stride-1 loops.  Adding register
   tiles reduces the loop length, reducing vectorisation opportunity.

3. **Hardware OOO and store forwarding compensate for C traffic on modern
   CPUs.**  On x86-64 with smaller OOO windows this matters more; on Apple M
   chips, less so.

4. **The BLAS micro-kernel combines register blocking and SIMD as one.**  They
   are not sequential steps; the register tile *is* the SIMD tile.

5. **Benchmarking is part of the process.**  The wrong loop structure gives a
   measured regression.  Measuring before claiming a speedup is mandatory.

---

## Exercises

1. Compile with `-O2 -fno-tree-vectorize` and re-run both `BM_MatmulBlocked`
   and `BM_MatmulRegBlocked`.  How does the gap change without auto-vectorisation?

2. Try `reg_size = 1` (degenerate case: register tile = 1 scalar).  Is it
   faster or slower than `reg_size = 4`?  Explain in terms of loop overhead.

3. Remove the `c[4][4]` load/write-back and instead accumulate directly into
   C (like `matmul_blocked`) but keep the `ir/jr` loop structure.  What do you
   expect vs what do you measure?

4. Draw the dependency graph for the 4 FMA chains in the BLAS micro-kernel
   sketch in Sec.7.  How many independent chains are there?  Why does that number
   equal the register tile height?

5. *(Advanced)* Measure with `perf stat -e fp_arith_inst_retired.256b_packed_double`.
   Does `matmul_blocked` emit more packed FP instructions than
   `matmul_register_blocked` despite doing the same number of FLOPs?

---

## References

- Goto, K. & van de Geijn, R. (2008). *Anatomy of high-performance matrix
  multiplication.* ACM TOMS 34(3). -- Sec.4 shows the register tile as the SIMD
  micro-kernel; Sec.5 explains why register blocking alone is not sufficient.
- van Zee, F. & van de Geijn, R. (2015). *BLIS: A framework for rapidly
  instantiating BLAS functionality.* ACM TOMS 41(3). -- The BLIS framework
  makes the register-tile / cache-tile split explicit in its object model.
- Fog, A. (2023). *Optimizing software in C++*. agner.org/optimize. -- Sec.12
  covers register allocation limits and when arrays stay on the stack vs
  spill to memory.
