# Week 8: Explicit SIMD -- AVX-256 and ARM NEON {#page_week8}

## Overview

Last week we saw that register blocking in scalar C++ showed a **regression**
because shrinking the inner j-loop from 64 to 4 iterations broke
auto-vectorisation.  The correct fix is to make those 4 iterations *one vector
instruction* -- processing all 4 elements simultaneously.  That is what explicit
SIMD delivers.

This week we implement two backends -- AVX-256 for x86-64, ARM NEON for
AArch64 -- behind a single `matmul_simd` / `matvec_simd` API, selected at
compile time.

Results on Apple M (ARM NEON, 2 doubles/register):

| N | Naive | Cache-blocked | SIMD (NEON) | vs Naive | vs Cache-blocked |
|---|---|---|---|---|---|
| 64 | 161 mus | 32 mus | **19 mus** | **8.3x** | **1.7x** |
| 128 | 1849 mus | 263 mus | **154 mus** | **12.0x** | **1.7x** |
| 256 | 18214 mus | 2756 mus | **1343 mus** | **13.6x** | **2.1x** |
| 512 | 159553 mus | 24793 mus | **11498 mus** | **13.9x** | **2.2x** |

BigO coefficient: **1.19 -> 0.18 -> 0.09 N^3** -- another 2x from SIMD alone.

---

## 1. SIMD Fundamentals

**SIMD** (Single Instruction, Multiple Data) executes one instruction on a
vector of elements simultaneously.  The two relevant ISA extensions:

| ISA | Register width | Doubles per register | Instruction set |
|---|---|---|---|
| AVX-256 (x86-64) | 256 bits | **4** | `_mm256_*` (`<immintrin.h>`) |
| ARM NEON (AArch64) | 128 bits | **2** | `v*q_f64` (`<arm_neon.h>`) |
| AVX-512 (x86-64) | 512 bits | **8** | `_mm512_*` |

On x86-64, the 256-bit registers are named `YMM0-YMM15` (16 registers).
On AArch64, the 128-bit SIMD registers are named `V0-V31` (32 registers).

The key instruction in both cases is **fused multiply-add (FMA)**:

```
AVX-256:  _mm256_fmadd_pd(a, b, c)   ->  c[0..3] += a[0..3] * b[0..3]
NEON:     vfmaq_f64(c, a, b)         ->  c[0..1] += a[0..1] * b[0..1]
```

One FMA instruction replaces what would be a 4-iteration (or 2-iteration)
scalar loop -- at no additional latency cost.

---

## 2. The Micro-kernel

### Why register blocking + SIMD work together

From week 7: register blocking's 4-iteration j-loop in scalar code is too
small to vectorise effectively.  The same 4-iteration loop maps to **one**
`_mm256_fmadd_pd` (AVX) or **two** `vfmaq_f64` (NEON) instructions.  Now the
"loop" has zero iterations of overhead.

The 4x4 register tile processes one k-step as:

```
// One step of the k-loop (AVX-256 pseudocode):
b   = load B[k, jr..jr+3]              // 1 vmovupd     -- 4 doubles
c0 += broadcast(A[ir+0, k]) * b        // 1 vbroadcastsd + 1 vfmadd
c1 += broadcast(A[ir+1, k]) * b        // 1 vbroadcastsd + 1 vfmadd
c2 += broadcast(A[ir+2, k]) * b        // 1 vbroadcastsd + 1 vfmadd
c3 += broadcast(A[ir+3, k]) * b        // 1 vbroadcastsd + 1 vfmadd
```

4 independent FMAs per k-step.  With two FMA execution ports on modern x86
CPUs (Skylake and later), the CPU can issue 2 FMAs per cycle -> theoretical
throughput of 2 x 4 = **8 doubles/cycle**.

For NEON, the 4x4 tile is split into two halves (lo = j..j+1, hi = j+2..j+3):

```
// One step of the k-loop (NEON pseudocode):
blo = load B[k, jr..jr+1]             // 2 doubles
bhi = load B[k, jr+2..jr+3]           // 2 doubles
c0lo += vdup(A[ir+0,k]) * blo         // vfmaq_f64
c0hi += vdup(A[ir+0,k]) * bhi         // vfmaq_f64
// ... x4 rows = 8 vfmaq_f64 per k-step
```

8 independent NEON FMAs per k-step, each processing 2 doubles.

### AVX-256 tile (`src/core/matrix_simd.cpp`)

```cpp
static inline void avx_tile_4x4(const Matrix& A, const Matrix& B, Matrix& C,
                                 idx ir, idx jr, idx kk, idx k_lim)
{
    const idx N = B.cols();
    real* Crow = C.data() + ir * N;

    // Load 4x4 C tile -- 4 YMM registers, no loop overhead
    __m256d c0 = _mm256_loadu_pd(Crow + 0 * N + jr);
    __m256d c1 = _mm256_loadu_pd(Crow + 1 * N + jr);
    __m256d c2 = _mm256_loadu_pd(Crow + 2 * N + jr);
    __m256d c3 = _mm256_loadu_pd(Crow + 3 * N + jr);

    for (idx k = kk; k < k_lim; ++k) {
        __m256d b  = _mm256_loadu_pd(B.data() + k * N + jr);   // 1 load
        c0 = _mm256_fmadd_pd(_mm256_set1_pd(A(ir+0, k)), b, c0); // independent
        c1 = _mm256_fmadd_pd(_mm256_set1_pd(A(ir+1, k)), b, c1); // independent
        c2 = _mm256_fmadd_pd(_mm256_set1_pd(A(ir+2, k)), b, c2); // independent
        c3 = _mm256_fmadd_pd(_mm256_set1_pd(A(ir+3, k)), b, c3); // independent
    }

    _mm256_storeu_pd(Crow + 0 * N + jr, c0);
    // ...
}
```

`c0..c3` are four independent YMM accumulators -- the CPU's OOO engine can
pipeline all four FMAs simultaneously.

### ARM NEON tile (`src/core/matrix_simd.cpp`)

```cpp
static inline void neon_tile_4x4(const Matrix& A, const Matrix& B, Matrix& C,
                                  idx ir, idx jr, idx kk, idx k_lim)
{
    // 4 rows x 2 NEON regs per row = 8 Q-registers for C
    float64x2_t c0lo = vld1q_f64(Crow + 0*N + jr);
    float64x2_t c0hi = vld1q_f64(Crow + 0*N + jr + 2);
    // ... x4 rows

    for (idx k = kk; k < k_lim; ++k) {
        float64x2_t blo = vld1q_f64(B.data() + k*N + jr);
        float64x2_t bhi = vld1q_f64(B.data() + k*N + jr + 2);
        float64x2_t a0  = vdupq_n_f64(A(ir+0, k));   // broadcast
        // ...
        c0lo = vfmaq_f64(c0lo, a0, blo);   // 8 independent FMAs per k step
        c0hi = vfmaq_f64(c0hi, a0, bhi);
        // ...
    }
}
```

AArch64 has 32 128-bit NEON registers.  The 4x4 tile uses 8 for C, 2 for B,
4 for A broadcasts = 14 total.  Plenty of headroom.

---

## 3. matvec_simd: Dot Product with Horizontal Reduction

Matrix-vector multiply is simpler: each row is a dot product.

### AVX-256 version

```cpp
for (idx i = 0; i < M; ++i) {
    __m256d acc = _mm256_setzero_pd();
    idx j = 0;

    for (; j + 4 <= N; j += 4) {
        __m256d a  = _mm256_loadu_pd(A.data() + i * N + j);
        __m256d xv = _mm256_loadu_pd(x.data() + j);
        acc = _mm256_fmadd_pd(a, xv, acc);    // acc[0..3] += a[0..3] * x[0..3]
    }

    // Horizontal sum: acc[0]+acc[1]+acc[2]+acc[3]
    __m128d lo  = _mm256_castpd256_pd128(acc);
    __m128d hi  = _mm256_extractf128_pd(acc, 1);
    __m128d sum = _mm_add_pd(lo, hi);
    sum = _mm_hadd_pd(sum, sum);
    y[i] = _mm_cvtsd_f64(sum) + scalar_tail;
}
```

The horizontal reduction collapses 4 partial sums into 1 at the end of each
row -- one reduction per row, not per element.

### NEON version

```cpp
float64x2_t acc = vdupq_n_f64(0.0);
for (; j + 2 <= N; j += 2) {
    acc = vfmaq_f64(acc, vld1q_f64(A.data() + i*N + j),
                         vld1q_f64(x.data() + j));
}
// Horizontal sum: acc[0] + acc[1]
real result = vgetq_lane_f64(acc, 0) + vgetq_lane_f64(acc, 1);
```

`vgetq_lane_f64` extracts each lane -- two scalar adds instead of a full
shuffle sequence.

### matvec benchmark results

| N | Scalar matvec (CPU) | SIMD matvec | Speedup |
|---|---|---|---|
| 64 | 1469 ns (21.4 GB/s) | 829 ns (38.0 GB/s) | **1.8x** |
| 128 | 8288 ns (15.0 GB/s) | 4641 ns (26.7 GB/s) | **1.8x** |
| 256 | 34741 ns (14.2 GB/s) | 24035 ns (20.5 GB/s) | **1.4x** |
| 512 | 191685 ns (10.2 GB/s) | 114969 ns (17.1 GB/s) | **1.7x** |
| 2048 | 3744813 ns (8.4 GB/s) | 2445977 ns (12.8 GB/s) | **1.5x** |

matvec is memory-bandwidth-bound (every element of A is read once), so the
SIMD speedup is modest (~1.5-1.8x) -- the bottleneck shifts from compute to
memory bus, not from SIMD efficiency.

---

## 4. Compile-Time Dispatch

### Architecture detection (`CMakeLists.txt`)

```cmake
include(CheckCXXCompilerFlag)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i686")
    check_cxx_compiler_flag("-mavx2 -mfma" COMPILER_SUPPORTS_AVX2)
    if(COMPILER_SUPPORTS_AVX2)
        target_compile_options(numerics PUBLIC -mavx2 -mfma)
        target_compile_definitions(numerics PUBLIC NUMERICS_HAS_AVX2)
    endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64|ARM64|AARCH64")
    target_compile_definitions(numerics PUBLIC NUMERICS_HAS_NEON)
endif()
```

`-mavx2 -mfma` tells the compiler to emit YMM instructions and enables FMA3
instruction encoding.  Without `-mfma`, `_mm256_fmadd_pd` compiles to
separate MUL + ADD instructions -- two instructions instead of one.

On AArch64, NEON is mandatory and always enabled; no flags needed.

### Dispatch function (`src/core/matrix_simd.cpp`)

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

The fallback ensures the library compiles and runs correctly on any
architecture -- important for CI/CD on heterogeneous build farms.

---

## 5. Boundary Handling

The tile functions process complete 4x4 blocks.  When M or N is not a
multiple of 4, boundary rows and columns fall through to scalar loops:

```cpp
idx ir = ii;
for (; ir + 4 <= i_lim; ir += 4) {     // <- full rows
    idx jr = jj;
    for (; jr + 4 <= j_lim; jr += 4)   // <- full 4-wide tiles
        simd_tile(A, B, C, ir, jr, kk, k_lim);
    for (; jr < j_lim; ++jr)           // <- remainder columns: scalar
        ...
}
for (; ir < i_lim; ++ir)               // <- remainder rows: scalar
    ...
```

For benchmark sizes that are powers of 2 (64, 128, 256, 512) and a tile
width of 4, the boundary path is never reached -- all elements fall into full
tiles.  For production use, the scalar tail handles all cases correctly with
no padding required.

---

## 6. Complete Benchmark Results

Measured on Apple M (AArch64, NEON), `-O2`, single-threaded.

```
Benchmark                            Time        Complexity
------------------------------------------------------------
BM_Matmul_CPU/64               161 mus
BM_Matmul_CPU/512          159553 mus
BM_Matmul_CPU_BigO              1.19 N^3   (baseline)

BM_MatmulBlocked_CPU/64         32 mus
BM_MatmulBlocked_CPU/512     24793 mus
BM_MatmulBlocked_CPU_BigO       0.18 N^3   (6.6x vs naive)

BM_MatmulSIMD_CPU/64            19 mus
BM_MatmulSIMD_CPU/128          154 mus
BM_MatmulSIMD_CPU/256         1343 mus
BM_MatmulSIMD_CPU/512        11498 mus
BM_MatmulSIMD_CPU_BigO          0.09 N^3   (13.9x vs naive, 2.2x vs blocked)
```

### Progression table

```
Naive i-j-k             1.19 N^3  (baseline)
+ cache blocking         0.18 N^3  (6.6x gain)
+ SIMD (NEON)            0.09 N^3  (2.2x gain on top -- 13.9x total)
---------------------------------------------------------------------
OpenBLAS equivalent     ~0.005 N^3  (remaining gap: ~18x)
```

The remaining gap to OpenBLAS is closed by:
1. **Wider tiles** -- NEON's 2-doubles-wide register is half of AVX-256's 4;
   on x86 the gains would be proportionally larger.
2. **B panel packing** -- packing the B tile into a contiguous buffer before
   the k-loop eliminates stride penalties and enables prefetch.
3. **Software prefetch** -- `__builtin_prefetch` hints load the next B tile
   into L1 while the CPU is computing the current one.
4. **Multithreading** -- OpenMP `parallel for` on the `ii` loop.

---

## 7. Key Takeaways

1. **SIMD turns a 4-iteration loop into one instruction.**  The register tile
   from week 7 was the right structure; it just needed vector width to pay off.

2. **AVX-256 and NEON have the same tile structure, different widths.**
   AVX processes 4 doubles per FMA; NEON processes 2.  The code is identical
   except for the intrinsic names and register counts.

3. **Compile-time dispatch keeps the API clean.**  Users call `matmul_simd`
   and get the best available backend without `#ifdef` scattered through
   application code.

4. **matvec is bandwidth-bound, not compute-bound.**  SIMD gives ~1.6x here
   because the bottleneck is memory bandwidth, not FLOPs.  No amount of
   vectorisation can exceed memory bandwidth.

5. **FMA requires an explicit compiler flag on x86.**  `-mfma` enables the
   fused instruction encoding.  Omitting it causes the compiler to emit
   separate MUL + ADD -- same FLOP count, 2x instruction count.

---

## 8. Exercises

1. On x86, compile `matmul_simd` with and without `-mfma`.  Disassemble the
   inner loop with `objdump -d`.  Count `vfmadd` instructions vs `vmul` +
   `vadd` pairs.

2. Try a 2x8 register tile for AVX (2 rows, 8 columns = 2 YMM regs per row).
   Does it beat the 4x4 tile?  What limits the tile height?

3. Add a `matmul_simd` benchmark with `block_size = 32`.  Does L1 vs L2
   residency change the result now that the inner loop is SIMD-accelerated?

4. Implement a SIMD `dot` for `Vector` using the same horizontal-reduce
   pattern from `matvec_simd`.  Benchmark against the scalar `dot`.

5. *(Advanced)* Add `#pragma omp parallel for` to the outer `ii` loop of
   `matmul_simd`.  What speedup do you observe on 4 cores?  Is it linear?
   What limits scaling?

---

## References

- Intel Intrinsics Guide: [https://www.intel.com/content/www/us/en/docs/intrinsics-guide/](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/) -- searchable reference for all `_mm*` functions.
- ARM NEON Intrinsics Reference: [https://arm-software.github.io/acle/neon_intrinsics/advsimd.html](https://arm-software.github.io/acle/neon_intrinsics/advsimd.html)
- Goto, K. & van de Geijn, R. (2008). *Anatomy of high-performance matrix multiplication.* ACM TOMS 34(3). -- The canonical paper; Sec.4 is the SIMD micro-kernel.
- Fog, A. (2023). *Optimizing software in C++.* agner.org/optimize. -- Sec.13 covers SIMD programming on x86; Sec.15 covers FMA throughput and latency.
