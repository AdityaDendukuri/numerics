# num::kernel Reference {#page_kernel}

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

## All routines

Grouped by what each does. All 97 are templates on the scalar type, take raw pointers
and lengths, allocate nothing, and are `noexcept`.

<div class="sym-index">
<div class="kidx-group"><span class="kidx-title">Vector construction <span class="hdr">&lt;kernel/vector.hpp&gt;</span></span><br/><span class="kidx-syms">@ref num::kernel::copy "copy" &ndash; @ref num::kernel::fill "fill" &ndash; @ref num::kernel::copy_strided "copy_strided" &ndash; @ref num::kernel::scale_copy_strided "scale_copy_strided" &ndash; @ref num::kernel::swap "swap" &ndash; @ref num::kernel::swap_strided "swap_strided"</span></div><div class="kidx-group"><span class="kidx-title">Vector arithmetic <span class="hdr">&lt;kernel/vector.hpp&gt;</span></span><br/><span class="kidx-syms">@ref num::kernel::scale "scale" &ndash; @ref num::kernel::axpy "axpy" &ndash; @ref num::kernel::axpy_strided "axpy_strided" &ndash; @ref num::kernel::axpby "axpby" &ndash; @ref num::kernel::axpbyz "axpbyz" &ndash; @ref num::kernel::add "add" &ndash; @ref num::kernel::hadamard_mul "hadamard_mul" &ndash; @ref num::kernel::hadamard_div "hadamard_div" &ndash; @ref num::kernel::inv "inv" &ndash; @ref num::kernel::clamp "clamp"</span></div><div class="kidx-group"><span class="kidx-title">Reductions <span class="hdr">&lt;kernel/vector.hpp&gt;</span></span><br/><span class="kidx-syms">@ref num::kernel::dot "dot" &ndash; @ref num::kernel::sum "sum" &ndash; @ref num::kernel::norm "norm" &ndash; @ref num::kernel::norm_sq "norm_sq" &ndash; @ref num::kernel::norm_sq_strided "norm_sq_strided" &ndash; @ref num::kernel::l1_norm "l1_norm" &ndash; @ref num::kernel::linf_norm "linf_norm" &ndash; @ref num::kernel::argmax_abs "argmax_abs"</span></div><div class="kidx-group"><span class="kidx-title">Fused reductions <span class="hdr">&lt;kernel/vector.hpp&gt;</span></span><br/><span class="kidx-syms">@ref num::kernel::axpy_norm_sq "axpy_norm_sq" &ndash; @ref num::kernel::dot2 "dot2" &ndash; @ref num::kernel::dot_norm_sq "dot_norm_sq" &ndash; @ref num::kernel::linear_combination_norm_sq "linear_combination_norm_sq"</span></div><div class="kidx-group"><span class="kidx-title">Matrix-vector products <span class="hdr">&lt;kernel/dense.hpp&gt;</span></span><br/><span class="kidx-syms">@ref num::kernel::matvec "matvec" &ndash; @ref num::kernel::matvec_transpose "matvec_transpose" &ndash; @ref num::kernel::gbmv "gbmv" &ndash; @ref num::kernel::ger "ger"</span></div><div class="kidx-group"><span class="kidx-title">Matrix-matrix products <span class="hdr">&lt;kernel/dense.hpp&gt;</span></span><br/><span class="kidx-syms">@ref num::kernel::gemm "gemm" &ndash; @ref num::kernel::gemm_config "gemm_config" &ndash; @ref num::kernel::gemm_workspace "gemm_workspace" &ndash; @ref num::kernel::gemm_transpose_left "gemm_transpose_left" &ndash; @ref num::kernel::syrk_lower "syrk_lower" &ndash; @ref num::kernel::transpose "transpose"</span></div><div class="kidx-group"><span class="kidx-title">Triangular solves <span class="hdr">&lt;kernel/dense.hpp&gt;</span></span><br/><span class="kidx-syms">@ref num::kernel::trsv_lower "trsv_lower" &ndash; @ref num::kernel::trsv_upper "trsv_upper" &ndash; @ref num::kernel::trsv_lower_inplace "trsv_lower_inplace" &ndash; @ref num::kernel::trsv_upper_inplace "trsv_upper_inplace" &ndash; @ref num::kernel::trsv_transpose_lower "trsv_transpose_lower" &ndash; @ref num::kernel::trsv_transpose_upper "trsv_transpose_upper" &ndash; @ref num::kernel::trsm_lower_inplace "trsm_lower_inplace" &ndash; @ref num::kernel::trsm_unit_lower_inplace "trsm_unit_lower_inplace" &ndash; @ref num::kernel::trsm_lower_transpose_inplace "trsm_lower_transpose_inplace" &ndash; @ref num::kernel::trsm_unit_lower_transpose_inplace "trsm_unit_lower_transpose_inplace" &ndash; @ref num::kernel::trsm_upper_inplace "trsm_upper_inplace" &ndash; @ref num::kernel::trsm_upper_transpose_inplace "trsm_upper_transpose_inplace" &ndash; @ref num::kernel::trsm_lower_transpose_right_inplace "trsm_lower_transpose_right_inplace"</span></div><div class="kidx-group"><span class="kidx-title">Orthogonalization <span class="hdr">&lt;kernel/dense.hpp&gt;</span></span><br/><span class="kidx-syms">@ref num::kernel::mgs_columns "mgs_columns" &ndash; @ref num::kernel::project_columns "project_columns" &ndash; @ref num::kernel::column_dot "column_dot" &ndash; @ref num::kernel::combine_columns "combine_columns" &ndash; @ref num::kernel::rotate_columns "rotate_columns" &ndash; @ref num::kernel::swap_rows "swap_rows"</span></div><div class="kidx-group"><span class="kidx-title">LU without pivoting <span class="hdr">&lt;kernel/dense.hpp&gt;</span></span><br/><span class="kidx-syms">@ref num::kernel::lu_no_pivot "lu_no_pivot" &ndash; @ref num::kernel::lu_no_pivot_solve_multiple "lu_no_pivot_solve_multiple" &ndash; @ref num::kernel::lu_no_pivot_solve_transpose_multiple "lu_no_pivot_solve_transpose_multiple"</span></div><div class="kidx-group"><span class="kidx-title">Cholesky <span class="hdr">&lt;kernel/factor.hpp&gt;</span></span><br/><span class="kidx-syms">@ref num::kernel::cholesky "cholesky" &ndash; @ref num::kernel::cholesky_blocked "cholesky_blocked" &ndash; @ref num::kernel::cholesky_solve "cholesky_solve" &ndash; @ref num::kernel::cholesky_batched "cholesky_batched" &ndash; @ref num::kernel::cholesky_solve_batched "cholesky_solve_batched" &ndash; @ref num::kernel::cholesky_invert "cholesky_invert"</span></div><div class="kidx-group"><span class="kidx-title">LU with partial pivoting <span class="hdr">&lt;kernel/factor.hpp&gt;</span></span><br/><span class="kidx-syms">@ref num::kernel::lu_factor "lu_factor" &ndash; @ref num::kernel::lu_factor_blocked "lu_factor_blocked" &ndash; @ref num::kernel::lu_solve "lu_solve" &ndash; @ref num::kernel::lu_invert "lu_invert"</span></div><div class="kidx-group"><span class="kidx-title">Banded <span class="hdr">&lt;kernel/factor.hpp&gt;</span></span><br/><span class="kidx-syms">@ref num::kernel::banded_factor "banded_factor" &ndash; @ref num::kernel::banded_solve "banded_solve"</span></div><div class="kidx-group"><span class="kidx-title">Givens and Jacobi rotations <span class="hdr">&lt;kernel/rotations.hpp&gt;</span></span><br/><span class="kidx-syms">@ref num::kernel::rotg "rotg" &ndash; @ref num::kernel::rot "rot" &ndash; @ref num::kernel::jacobi_rotation "jacobi_rotation"</span></div><div class="kidx-group"><span class="kidx-title">Householder and QR <span class="hdr">&lt;kernel/rotations.hpp&gt;</span></span><br/><span class="kidx-syms">@ref num::kernel::householder_vector "householder_vector" &ndash; @ref num::kernel::householder_vector_strided "householder_vector_strided" &ndash; @ref num::kernel::householder_left "householder_left" &ndash; @ref num::kernel::householder_right "householder_right" &ndash; @ref num::kernel::qr_form_block "qr_form_block" &ndash; @ref num::kernel::qr_apply_block_left "qr_apply_block_left" &ndash; @ref num::kernel::qr_factor_blocked "qr_factor_blocked" &ndash; @ref num::kernel::qr_workspace "qr_workspace"</span></div><div class="kidx-group"><span class="kidx-title">Sparse products <span class="hdr">&lt;kernel/sparse.hpp&gt;</span></span><br/><span class="kidx-syms">@ref num::kernel::spmv "spmv" &ndash; @ref num::kernel::spmv_axpy "spmv_axpy" &ndash; @ref num::kernel::spmm "spmm"</span></div><div class="kidx-group"><span class="kidx-title">Incomplete factorization <span class="hdr">&lt;kernel/sparse.hpp&gt;</span></span><br/><span class="kidx-syms">@ref num::kernel::csr_diagonal_positions "csr_diagonal_positions" &ndash; @ref num::kernel::ilu0_factor "ilu0_factor" &ndash; @ref num::kernel::csr_lu_solve "csr_lu_solve"</span></div><div class="kidx-group"><span class="kidx-title">Krylov <span class="hdr">&lt;kernel/krylov.hpp&gt;</span></span><br/><span class="kidx-syms">@ref num::kernel::cg "cg" &ndash; @ref num::kernel::pcg "pcg"</span></div><div class="kidx-group"><span class="kidx-title">Real and complex <span class="hdr">&lt;kernel/complex.hpp&gt;</span></span><br/><span class="kidx-syms">@ref num::kernel::matvec_real_complex "matvec_real_complex" &ndash; @ref num::kernel::matvec_transpose_into_complex "matvec_transpose_into_complex" &ndash; @ref num::kernel::hessenberg_shifted_factor "hessenberg_shifted_factor" &ndash; @ref num::kernel::hessenberg_shifted_substitute "hessenberg_shifted_substitute" &ndash; @ref num::kernel::hessenberg_shifted_solve "hessenberg_shifted_solve"</span></div>
</div>

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

## 3. Vector operations

`<kernel/vector.hpp>`

| Signature | Effect | Complexity |
| :--- | :--- | :--- |
| `copy(y, x, n)` | \f$y_i \leftarrow x_i\f$ | 2n |
| `fill(x, value, n)` | \f$x_i \leftarrow \mathit{value}\f$ | n |
| `scale(x, alpha, n)` | \f$x_i \leftarrow \alpha x_i\f$ | 2n |
| `axpy(y, x, alpha, n)` | \f$y_i \leftarrow y_i + \alpha x_i\f$ | 3n |
| `axpby(y, x, a, b, n)` | \f$y_i \leftarrow a x_i + b y_i\f$ | 3n |
| `axpbyz(z, x, y, a, b, n)` | \f$z_i \leftarrow a x_i + b y_i\f$ | 3n |
| `add(z, x, y, n)` | \f$z_i \leftarrow x_i + y_i\f$ | 3n |
| `hadamard_mul(z, x, y, n)` | \f$z_i \leftarrow x_i y_i\f$ | 3n |
| `hadamard_div(z, x, y, n)` | \f$z_i \leftarrow x_i / y_i\f$ | 3n |
| `inv(y, x, n)` | \f$y_i \leftarrow 1/x_i\f$ | 2n |
| `clamp(x, lo, hi, n)` | \f$x_i \leftarrow \min(\max(x_i, lo), hi)\f$ | 2n |
| `swap(x, y, n)` | \f$x \leftrightarrow y\f$ | 4n |
| `dot(x, y, n)` | \f$\sum_i x_i y_i\f$ | 2n |
| `sum(x, n)` | \f$\sum_i x_i\f$ | n |
| `norm(x, n)` | \f$\Vert x \Vert_2\f$ | n |
| `norm_sq(x, n)` | \f$\Vert x \Vert_2^2\f$ | n |
| `l1_norm(x, n)` | \f$\Vert x \Vert_1\f$ | n |
| `linf_norm(x, n)` | \f$\max_i \lvert x_i \rvert\f$ | n |
| `argmax_abs(x, n)` | \f$\arg\max_i \lvert x_i \rvert\f$ | n |

Strided variants take an increment after each pointer: `copy_strided`,
`scale_copy_strided`, `axpy_strided`, `norm_sq_strided`, `swap_strided`.

### num::kernel::axpy

```cpp
template <std::floating_point T>
void axpy(T *y, const T *x, T alpha, idx n) noexcept;
```

Computes \f$y \leftarrow y + \alpha x\f$ in place.

| Parameter | Meaning |
| :--- | :--- |
| `y` | Length `n`, read and written. Must not alias `x`. |
| `x` | Length `n`, read only. |
| `alpha` | Scalar multiplier. |
| `n` | Number of elements. `n == 0` is permitted and does nothing. |

### num::kernel::dot

```cpp
template <std::floating_point T>
[[nodiscard]] T dot(const T *x, const T *y, idx n) noexcept;
template <std::floating_point T>
[[nodiscard]] T dot(contract::ordered_t, const T *x, const T *y, idx n) noexcept;
```

Returns \f$\sum_i x_i y_i\f$. The first overload accumulates into several independent
chains and does not sum in source order. The second sums strictly in increasing index
order. See §6 for which to use.

### num::kernel::norm

```cpp
template <std::floating_point T>
[[nodiscard]] T norm(const T *x, idx n) noexcept;
```

Returns \f$\Vert x \Vert_2\f$.

Computed as `sqrt(norm_sq(x, n))` when that result is finite and non-zero, and by a
rescaled second pass otherwise. The direct form squares before it sums, so it overflows to
infinity once \f$\Vert x \Vert_2\f$ exceeds about `1.3e154` in double precision, and
flushes to zero below about `1.5e-154`. Both cases return a wrong answer for a vector
whose every element is an ordinary finite number. Only those cases pay for the second
pass.

### Fused reductions

These traverse their operands once and produce both an update and a statistic. In an
iterative method the updated values are still in registers when the reduction consumes
them, so the fused form is close to bandwidth-optimal.

| Signature | Returns | Also does |
| :--- | :--- | :--- |
| `axpy_norm_sq(y, x, alpha, n)` | \f$\Vert y + \alpha x \Vert^2\f$ | \f$y \leftarrow y + \alpha x\f$ |
| `dot2(x, y, z, n)` | `{xy, xz}` | Loads `x` once for both products |
| `dot_norm_sq(x, y, n)` | `{dot, norm_sq}` | One pass over both |
| `linear_combination_norm_sq(x, a, y, b, n)` | \f$\Vert a x + b y \Vert^2\f$ | Does not materialize the combination |

---

## 4. Dense operations

`<kernel/dense.hpp>`

| Signature | Effect |
| :--- | :--- |
| `matvec(y, A, x, m, n)` | \f$y \leftarrow Ax\f$ |
| `matvec_transpose(y, A, x, m, n)` | \f$y \leftarrow A^T x\f$ |
| `gemm(C, A, B, alpha, beta, m, n, k)` | \f$C \leftarrow \alpha AB + \beta C\f$ |
| `gemm(C, ldc, A, lda, B, ldb, alpha, beta, m, n, k)` | The same, with explicit leading dimensions |
| `gemm(C, ldc, A, lda, B, ldb, alpha, beta, m, n, k, work)` | The same, packing into caller-provided workspace of `gemm_workspace<T>(m, n, k)` elements |
| `gemm_transpose_left(C, ldc, A, lda, B, ldb, ...)` | \f$C \leftarrow \alpha A^T B + \beta C\f$ |
| `syrk_lower(C, ldc, A, lda, alpha, beta, n, k)` | \f$C \leftarrow \alpha A A^T + \beta C\f$, lower triangle |
| `ger(A, lda, x, y, alpha, m, n)` | \f$A \leftarrow A + \alpha x y^T\f$ |
| `gbmv(y, A, x, m, n, kl, ku)` | Banded matrix-vector product |
| `transpose(B, A, m, n)` | \f$B \leftarrow A^T\f$ |
| `trsv_lower`, `trsv_upper`, and transposed and in-place variants | Triangular solve, one right-hand side |
| `trsm_lower_inplace`, `trsm_lower_transpose_inplace`, `trsm_upper_inplace`, `trsm_upper_transpose_inplace`, unit-diagonal and right-side variants | Triangular solve, many right-hand sides. Blocked: diagonal blocks by substitution, the rest by `gemm`; the upper forms are the lower ones with strides swapped |
| `mgs_columns`, `project_columns`, `column_dot`, `combine_columns`, `rotate_columns` | Modified Gram-Schmidt building blocks |
| `lu_no_pivot(A, n)` | LU without pivoting, for matrices whose structure guarantees non-zero pivots |

### num::kernel::gemm

```cpp
template <std::floating_point T>
void gemm(T *C, idx ldc, const T *A, idx lda, const T *B, idx ldb,
          T alpha, T beta, idx m, idx n, idx k) noexcept;
```

Computes \f$C \leftarrow \alpha A B + \beta C\f$ for \f$A\f$ of `m` by `k`, \f$B\f$ of `k`
by `n`, and \f$C\f$ of `m` by `n`.

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

## 5. Sparse, factorizations, rotations, Krylov

| Header | Signature | Effect |
| :--- | :--- | :--- |
| sparse | `spmv(y, val, row_ptr, col_idx, x, m)` | \f$y \leftarrow Ax\f$ for CSR \f$A\f$ |
| sparse | `spmv_axpy(y, alpha, val, row_ptr, col_idx, x, beta, m)` | \f$y \leftarrow \alpha Ax + \beta y\f$ |
| sparse | `spmm(Y, ldy, val, row_ptr, col_idx, X, ldx, m, nrhs)` | \f$Y \leftarrow AX\f$ for a dense block \f$X\f$ |
| sparse | `csr_diagonal_positions(diagonal, row_ptr, col_idx, n)` | Locates each row's diagonal. Returns `false` if a row has none. |
| sparse | `ilu0_factor(val, row_ptr, col_idx, diagonal, scratch, n)` | ILU(0) in place. Returns `false` on a zero or non-finite pivot. |
| sparse | `csr_lu_solve(x, val, row_ptr, col_idx, diagonal, b, n)` | Solves \f$LUx = b\f$ for factors from `ilu0_factor`. `x` may alias `b`. |
| factor | `cholesky(L, A, n)` | \f$A = LL^T\f$. Returns `false` if `A` is not positive definite. |
| factor | `cholesky_solve(x, L, b, n)` | Solves from a Cholesky factor |
| factor | `cholesky_blocked`, `cholesky_batched`, `cholesky_solve_batched`, `cholesky_invert` | Blocked, batched and inverting variants |
| factor | `lu_factor(A, ipiv, n)` | \f$PA = LU\f$ in place. Returns `false` if singular. |
| factor | `lu_solve(x, LU, ipiv, b, n)` | Solves from an LU factor |
| factor | `banded_factor`, `banded_solve` | Banded LU |
| rotations | `rotg(a, b, c, s)` / `rot(x, y, c, s, n)` | Construct and apply a Givens rotation |
| rotations | `householder_vector(v, beta, x, n)` | Elementary reflector \f$(v, \beta)\f$ |
| rotations | `householder_left`, `householder_right` | Apply \f$I - \beta v v^T\f$ from either side |
| rotations | `qr_form_block`, `qr_apply_block_left` | Compact-WY block reflector \f$I - V T V^T\f$ from a factored panel, and its application by two `gemm`s |
| rotations | `qr_factor_blocked(A, lda, m, n, tau, work)` | Blocked compact Householder QR: panels of `qr_block` columns, trailing update through the block reflector; reflector tails stay below the diagonal of \f$R\f$; `work` holds `qr_workspace(m, n)` elements |
| rotations | `jacobi_rotation` | One Jacobi sweep step |
| krylov | `cg(A, x, b, n, work, tol, max_iter)` | Conjugate gradients over a callable \f$A\f$ |
| krylov | `pcg(A, M, x, b, n, work, tol, max_iter)` | Preconditioned conjugate gradients |

### num::kernel::cg

```cpp
template <std::floating_point T, class MatVec>
[[nodiscard]] krylov_result<T> cg(MatVec &&A, T *x, const T *b, idx n, T *work,
                                  T tol = T(1e-10), idx max_iter = 1000);
```

Solves \f$Ax = b\f$ for symmetric positive definite \f$A\f$.

| Parameter | Meaning |
| :--- | :--- |
| `A` | Any callable with signature `void(const T *u, T *Au)`. It is never stored. |
| `x` | Length `n`. The initial guess on entry, the solution on exit. |
| `b` | Length `n`, read only. |
| `work` | Length `3*n`. Caller-provided; `cg` allocates nothing. |
| `tol` | Relative residual at which to stop. |
| `max_iter` | Iteration cap. |

Returns `krylov_result<T>` with `iterations`, `residual` and `converged`.

Because `A` is a callable rather than a matrix, this works equally on a dense matrix, a
CSR matrix, or a stencil evaluated on the fly. Positive definiteness is the caller's
responsibility; nothing here verifies it.

---

## 6. Summation order

`dot`, `sum`, `norm`, `norm_sq`, `l1_norm` and the fused reductions spread the range
across several accumulators and combine them pairwise at the end. This is a permutation of
the source order, and it matters in two ways.

A single accumulator carries a loop-carried floating-point dependency. Addition is not
associative, so a compiler may not break it, and the loop then runs at the latency of the
adder rather than its throughput. Measured here, that is three to four times slower, and
the gap persists past cache size because the loop is latency-bound rather than
bandwidth-bound.

Bounding the chain length also makes the error grow like \f$O(n/K)\f$ instead of
\f$O(n)\f$. On unstructured data the blocked order is therefore more accurate as well as
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

## 7. Vectorization

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

## 8. Using it on its own

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

## See also

* @ref page_architecture "Library Structure & Architecture" — where this tier sits and what may depend on it
* @ref page_performance "Performance" — measured throughput for these routines
* @ref page_concepts "Concepts & Invariants" — the typed layers that establish the preconditions this tier assumes
