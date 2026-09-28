---
scope: num::kernel
---

# num::kernel

`num::kernel` is the computational core of the library. Every routine operates on raw
pointers, lengths and strides. Nothing in it allocates, throws, or refers to `num::vec`,
`num::mat`, or any type above it.

It has no dependencies beyond the C++ standard library, so it can be copied out of this
project and used on its own. Every other tier of `numerics` reaches this one: a call to
`num::cg` on a `num::vec` ends in the same loops a direct call to `num::kernel::cg` would
run.

```cpp
#include <kernel/kernel.hpp>          // all of it
#include <kernel/dense.hpp>           // or one header at a time
```

---

Every routine is listed under [All kernels](reference/kernel.md), each with its own page.

---

## 1. The contract

These rules hold for every function in the tier and are not repeated per routine. A
function's own preconditions are stated in addition to these.

**Nothing is checked.** No dimension is validated, no pointer is tested against null, no
divisor is tested for zero. Violating a precondition is undefined behaviour, not an
exception. Establishing the preconditions is the job of the tiers above; this tier assumes
that already happened.

**Buffers are caller-allocated and caller-sized.** A kernel never allocates, never frees,
never resizes, and never retains a pointer after it returns. A parameter documented as
length `n` must be readable, and if it is an output, writable, for exactly `n` elements.

**Restrict-qualified pointers must not overlap.** Most parameters are marked
`NUM_K_RESTRICT`. Passing one buffer as two such parameters is undefined behaviour. It
does not produce a diagnostic; at `-O2` it produces wrong answers. Where a function
permits aliasing, its documentation says so.

**Every routine is `noexcept`.** A kernel has no failure it can report by throwing.
Routines that can fail numerically return `bool` or a result struct.

**Reductions are not in source order.** `dot`, `norm`, `sum` and the fused reductions
spread the range across several accumulators. See §6.

**Row-major storage.** A matrix of `m` rows and `n` columns occupies `m*n` contiguous
elements; entry `(i, j)` is at `A[i*n + j]`. Routines taking an explicit leading dimension
name it `lda`, `ldb` or `ldc`.

Complexity is quoted in elements touched rather than in floating-point operations, since
every routine here is bandwidth-bound at realistic sizes.

---

## 2. Headers

| Header | Contents |
| :--- | :--- |
| `<kernel/vector.hpp>` | BLAS-1 vector operations, fused reductions, the `NUM_K_*` macros. Every other header includes it. |
| `<kernel/dense.hpp>` | BLAS-2 and BLAS-3: `gemm`, `matvec`, triangular solves, banded products, Gram-Schmidt. |
| `<kernel/sparse.hpp>` | CSR products and ILU(0). |
| `<kernel/factor.hpp>` | Cholesky and LU, plain, blocked and batched. Banded solves. |
| `<kernel/rotations.hpp>` | Givens, Householder and Jacobi rotations. Blocked QR. |
| `<kernel/krylov.hpp>` | Matrix-free CG and PCG over a callable operator. |
| `<kernel/complex.hpp>` | Routines mixing real and complex operands. Kept separate because `<complex>` costs roughly 95,000 preprocessed lines. |
| `<kernel/debug.hpp>` | `operator<<` for `krylov_result`. The only kernel header that includes `<ostream>`, and deliberately not part of the umbrella. |

---

## 3. How gemm is blocked

```cpp
template <std::floating_point T>
void gemm(T *C, idx ldc, const T *A, idx lda, const T *B, idx ldb,
          T alpha, T beta, idx m, idx n, idx k) noexcept;
```

Computes $C \leftarrow \alpha A B + \beta C$ for $A$ of `m` by `k`, $B$ of `k`
by `n`, and $C$ of `m` by `n`.

| Parameter | Meaning |
| :--- | :--- |
| `C`, `ldc` | Output, `m` by `n`, leading dimension `ldc`. Read as well as written when `beta != 0`. |
| `A`, `lda` | Left operand, `m` by `k`. |
| `B`, `ldb` | Right operand, `k` by `n`. |
| `alpha`, `beta` | Scalars. `beta == 0` overwrites `C` without reading it, so uninitialized memory is acceptable there. |

`A`, `B` and `C` must not overlap.

The implementation is the Goto/BLIS structure, in portable C++:

1. **Packing.** Before any arithmetic, the slice of `A` in play is copied into an
   `mc x kc` slab laid out as `mr`-row tiles, and the slice of `B` into a `kc x nc` panel
   laid out as `nr`-column tiles. Both are contiguous and zero-padded to whole tiles, so
   the inner loop streams unit-stride memory and never sees a matrix edge; `alpha` is
   folded into the packed `A`.
2. **Three blocking loops** hold the `B` panel across one sweep of the `A` slab, the `A`
   slab in L2, and the `kc x nr` sliver of `B` the microkernel reads in L1.
3. **A register-tiled microkernel** computes one `mr x nr` block of `C` over `kc`
   products, its accumulators in `mr * nr / width` vector registers, written with the
   GCC/clang vector extension so the compiler emits one fused multiply-add per lane.

Every one of those integers comes from the target at compile time (`num::kernel::gemm_config`):
`mr x nr` from the vector width and register count, `kc`/`mc`/`nc` from `NUM_K_L1_BYTES`,
`NUM_K_L2_BYTES` and `NUM_K_GEMM_PANEL_BYTES`. The tile shapes this produces are the ones
BLIS ships per target:

| Target | `mr x nr` (double) | `kc` (32 KiB L1) |
| :--- | :--- | :--- |
| SSE2 | 6 x 4 | 512 |
| NEON | 8 x 6 | 512 |
| AVX2 | 6 x 8 | 256 |
| AVX-512 | 14 x 16 | 128 |

The cache macros default to the smallest sizes on any mainstream core, so a build that
knows nothing about its host is under-blocked but never wrong; numerics' CMake sets them
from the host's measured caches (attached to the `simd` backend target, so `numerics::kernel`
itself stays free of definitions). A consumer of the bare headers may define them before
including.

Per output element the summation is ascending in `p` within one `kc` panel and the panel
sums are added in order, so for `k <= kc` the result is bit-identical to a naive triple
loop compiled with the same contraction.

The overload without `work` packs into a per-thread static buffer of
`gemm_config<T>::workspace` elements. That is the one place in the tier with storage of
its own; it never calls an allocator and is safe from any thread. Pass `work` to keep the
call stateless. `syrk_lower` and `gemm_transpose_left` run through the same packed core,
the transposed operand expressed as a stride swap in the packer rather than a copy. So do
the six `trsm` forms (each diagonal block of `detail::trsm_block` rows by substitution,
the rows or columns beyond it by one `gemm`), and through them `cholesky_blocked`,
`lu_factor_blocked` and `qr_factor_blocked`.

Measured on an Apple M1 Pro (clang 17, one thread): `gemm` 44 GFLOP/s from n = 256 to
2048, which is 86% of the core's NEON FMA peak and within 10% of OpenBLAS's hand-written
NEON `dgemm` on the same core; Cholesky 35 and LU 28 GFLOP/s at n = 1024, from 6 and 9
before the blocking. Accelerate's `dgemm` measures 270-620 GFLOP/s on the same machine
because it runs on the AMX coprocessor, which no C++ can reach.

There are no `gemm_blocked`, `gemm_register_blocked` or `matmul_simd` variants. Those
existed and were removed: the blocked ones measured slower than this, and the hand-written
AVX2 and NEON products both measured slower *and* indexed `A` with the wrong leading
dimension, which made them silently wrong for any non-square shape.

---

## 4. Summation order

`dot`, `sum`, `norm`, `norm_sq`, `l1_norm` and the fused reductions spread the range
across several accumulators and combine them pairwise at the end. This is a permutation of
the source order, and it matters in two ways.

A single accumulator carries a loop-carried floating-point dependency. Addition is not
associative, so a compiler may not break it, and the loop then runs at the latency of the
adder rather than its throughput. Measured here, that is three to four times slower, and
the gap persists past cache size because the loop is latency-bound rather than
bandwidth-bound.

Bounding the chain length also makes the error grow like $O(n/K)$ instead of
$O(n)$. On unstructured data the blocked order is therefore more accurate as well as
faster. It is *less* accurate on data whose sign pattern is periodic in the accumulator
count, where source order cancels adjacent terms immediately.

The grouping is fixed for a given build but depends on the target's vector width. A result
computed under AVX-512 need not match the same source built for SSE bit for bit. Where a
result must reproduce a reference implementation exactly, or must be identical across
machines, use the `contract::ordered` overload.

```cpp
double fast  = num::kernel::dot(x, y, n);                            // blocked
double exact = num::kernel::dot(num::kernel::contract::ordered, x, y, n);  // source order
```

---

## 5. Vectorization

The kernel contains no intrinsics and performs no runtime CPU dispatch. It writes loops
the compiler can vectorize and blocks them for the register file and the cache. Two
compile-time constants describe the target's registers: `NUM_K_VECTOR_BYTES`, the vector
width, and `NUM_K_VECTOR_REGISTERS`, the size of the file; three describe its caches:
`NUM_K_L1_BYTES`, `NUM_K_L2_BYTES` and `NUM_K_GEMM_PANEL_BYTES`. `gemm` and the reductions
read those to size their tiles. `NUM_K_HAS_FMA` records whether `a*b + c` fuses on this
target: x86 needs `-mfma` (set with `-mavx2` by numerics' CMake), and GCC additionally
needs `-ffp-contract=fast`, which strict `-std=c++NN` turns off and numerics' CMake turns
back on. Without fusion the arithmetic is still correct and the ceiling halves.

Hand-written AVX2 and NEON paths existed here and were removed. On the same machine the
portable tiled `gemm` measured 30.0 GFLOP/s against their 23.7, and a hand-written
`matvec` intrinsic ran at 16.6 GiB/s against the portable `matvec`'s 49.1, because it
accumulated into one vector register and so ran at the latency of the FMA rather than its
throughput. Nothing in the tier now depends on a build flag, and nothing in it can raise
SIGILL on an older CPU.

---

## 6. Using it on its own

Copy `include/kernel/` into another project. It needs no build system, no linking, and no
other part of `numerics`. The headers carry an MIT licence notice and two attribution
lines; keep those with whatever you take.

```cpp
#include <kernel/kernel.hpp>
#include <cstdio>
#include <vector>

int main() {
    constexpr num::idx n = 4;
    std::vector<double> x{1.0, 2.0, 3.0, 4.0};
    std::vector<double> y{0.5, 1.5, 2.5, 3.5};

    const double d = num::kernel::dot(x.data(), y.data(), n);
    num::kernel::axpy(y.data(), x.data(), 2.0, n);   // y <- y + 2x

    std::printf("dot=%.1f y0=%.1f\n", d, y[0]);      // dot=25.0 y0=2.5
}
```

The kernel operates on whatever exposes contiguous storage, so `std::vector`,
`std::array`, a raw `new[]` buffer, an Eigen vector through `.data()`, or an Armadillo
matrix through `.memptr()` all work without adaptation.

A matrix-free solve, with the caller owning every buffer:

```cpp
#include <kernel/krylov.hpp>
#include <vector>

int main() {
    constexpr num::idx n = 1000;
    std::vector<double> b(n, 1.0), x(n, 0.0), work(3 * n);

    auto laplacian = [](const double *u, double *Lu) {
        for (num::idx i = 0; i < n; ++i) {
            Lu[i] = 2.0 * u[i] - (i > 0 ? u[i - 1] : 0.0)
                               - (i + 1 < n ? u[i + 1] : 0.0);
        }
    };

    auto r = num::kernel::cg(laplacian, x.data(), b.data(), n, work.data(), 1e-10, 2000);
    return r.converged ? 0 : 1;
}
```

`CMakeLists.txt` exposes the tier as its own target, so a project already using CMake can
depend on it without taking the rest:

```cmake
find_package(numerics REQUIRED COMPONENTS kernel)
target_link_libraries(my_program PRIVATE numerics::kernel)
```

---

