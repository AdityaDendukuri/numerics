/// @file kernel/dense.hpp
/// @brief Raw-pointer kernels: dense Level-2/3 BLAS, triangular solves, banded ops.
///
/// SPDX-License-Identifier: MIT
/// Part of numerics, (c) 2026 Aditya Dendukuri.
/// https://github.com/AdityaDendukuri/numerics
///
/// This file has no dependencies outside the standard library beyond
/// kernel/vector.hpp, whose macro block and NUM_K_* prefix it reuses: copy the
/// two into another project as-is, or lift a single routine out of it. Please
/// keep the two attribution lines above with whatever you take.
///
/// Kernels assume non-owning, caller-sized, row-major buffers and do not
/// allocate.
#pragma once

#include "kernel/vector.hpp"
#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstdlib>
#include <cstring>
#include <type_traits>

namespace num::kernel {

/// @brief Dense matrix-vector multiplication \f$\mathbf{y} \leftarrow A \mathbf{x}\f$ for \f$A \in
/// \mathbb{R}^{m \times n}\f$.
template <std::floating_point T>
NUM_K_AINLINE void matvec(T *NUM_K_RESTRICT y, const T *NUM_K_RESTRICT A, const T *NUM_K_RESTRICT x,
                          idx m, idx n) noexcept {
    for (idx i = 0; i < m; ++i) {
        const T *row = A + (i * n);
        y[i] = detail::reduce<T>(n, [row, x](idx j) { return row[j] * x[j]; });
    }
}

/// @brief Transposed dense matrix-vector multiplication \f$\mathbf{y} \leftarrow A^T \mathbf{x}\f$
/// for \f$A \in \mathbb{R}^{m \times n}\f$.
template <std::floating_point T>
NUM_K_AINLINE void matvec_transpose(T *NUM_K_RESTRICT y, const T *NUM_K_RESTRICT A,
                                    const T *NUM_K_RESTRICT x, idx m, idx n) noexcept {
    for (idx j = 0; j < n; ++j) {
        y[j] = T(0);
    }
    for (idx i = 0; i < m; ++i) {
        const T xi = x[i];
        const T *row = A + (i * n);
        NUM_K_IVDEP
        for (idx j = 0; j < n; ++j) {
            y[j] += row[j] * xi;
        }
    }
}

/// @brief Inner product of columns \f$p\f$ and \f$q\f$ of a row-major matrix.
template <std::floating_point T>
NUM_K_AINLINE T column_dot(const T *NUM_K_RESTRICT A, idx lda, idx rows, idx p, idx q) noexcept {
    return detail::reduce<T>(rows,
                             [A, lda, p, q](idx i) { return A[(i * lda) + p] * A[(i * lda) + q]; });
}

/// @brief Applies a Givens rotation to columns \f$p\f$ and \f$q\f$ in place.
template <std::floating_point T>
NUM_K_AINLINE void rotate_columns(T *NUM_K_RESTRICT A, idx lda, idx rows, idx p, idx q, T c,
                                  T s) noexcept {
    for (idx i = 0; i < rows; ++i) {
        T *row = A + (i * lda);
        const T ap = row[p], aq = row[q];
        row[p] = (c * ap) - (s * aq);
        row[q] = (s * ap) + (c * aq);
    }
}

// Dense matrix product.
//
// The shape of this is forced by the arithmetic intensity of the operation. A
// product does O(n^3) work over O(n^2) data, so it is compute-bound in
// principle, but the textbook i-k-j triple loop does not get anywhere near the
// machine's peak: each fused multiply-add reads a fresh element of C from
// memory and writes it straight back, so the loop runs at the rate the store
// unit and the L1 cache can retire traffic, not at the rate the FMA units can
// issue. On this tree the plain version sustained ~13 GFLOP/s.
//
// Two nested levels of blocking fix that, and nothing else is needed.
//
//   Register tile (`mr` x `nr`): the innermost loop holds a small block of C in
//   vector registers across the entire k sweep. Each element of A loaded is
//   reused across `nr` columns and each element of B across `mr` rows, so one
//   pair of loads feeds `mr*nr` FMAs instead of one. The tile is deliberately
//   sized to about half the architectural vector register file: large enough to
//   hide FMA latency, small enough that the accumulators are never spilled --
//   a spilled tile is slower than no tile at all.
//
//   Cache panel (`kc`): the k range is cut so the slice of B the tile loop
//   sweeps (kc x n) stays resident while every row block streams past it.
//   Without this, a large product re-reads B from DRAM once per row block.
//
// Both bounds come from the target's own properties, not from a tuning
// parameter, so there is nothing for a caller to get wrong. Together they take
// the same computation to ~2.5x the plain loop, and the summation order per
// output element is unchanged (still ascending in p), so results are
// bit-identical to the naive triple loop.

namespace detail {

[[nodiscard]] constexpr idx round_down(idx value, idx multiple) noexcept {
    return multiple == 0 ? value : (value / multiple) * multiple;
}
[[nodiscard]] constexpr idx round_up(idx value, idx multiple) noexcept {
    return multiple == 0 ? value : ((value + multiple - 1) / multiple) * multiple;
}

} // namespace detail

/// @brief The blocking `gemm` uses on this target, derived at compile time.
///
/// The structure is the Goto/BLIS one: a register-tiled microkernel computing an
/// `mr x nr` block of C over `kc` inner products, fed from packed copies of A and
/// B so that it streams contiguous, zero-padded panels rather than strided rows
/// of the caller's matrices. Three loops around it block for the memory
/// hierarchy: an `mc x kc` slab of A held in L2, a `kc x nc` panel of B held
/// across one sweep of that slab, and the `mr x kc` / `kc x nr` slivers the
/// microkernel touches held in L1.
///
/// Only five integers are target-specific, and every one is derived from the
/// macros in `kernel/vector.hpp`:
///
/// | | NEON / SSE2 | AVX2 | AVX-512 |
/// |---|---|---|---|
/// | `mr x nr` (double) | 8 x 6 / 6 x 4 | 6 x 8 | 14 x 16 |
///
/// which are the shapes BLIS ships for those targets. `kc`, `mc` and `nc` come
/// from `NUM_K_L1_BYTES`, `NUM_K_L2_BYTES` and `NUM_K_GEMM_PANEL_BYTES`.
template <std::floating_point T>
struct gemm_config {
#if defined(NUM_K_VECTOR_EXT)
    /// Elements per vector register.
    static constexpr idx width = NUM_K_VECTOR_BYTES / sizeof(T);
    /// B vectors held per microkernel: three when the file is wide and the
    /// vectors narrow (NEON), else two.
    static constexpr idx b_vectors = (NUM_K_VECTOR_REGISTERS >= 32 && width <= 2) ? 3 : 2;
    /// Rows of the register tile: what remains of the file after the B vectors
    /// and two broadcast registers, at most 14, rounded to even.
    static constexpr idx mr =
        std::min<idx>(14, ((NUM_K_VECTOR_REGISTERS - b_vectors - 2) / b_vectors) & ~idx{1});
#else
    static constexpr idx width = 4;
    static constexpr idx b_vectors = 2;
    static constexpr idx mr = 4;
#endif
    /// Columns of the register tile.
    static constexpr idx nr = b_vectors * width;
    /// Inner-product length per microkernel call. The `kc x nr` B sliver stays
    /// in L1 across the whole sweep of A tiles, so it gets half of L1; the A
    /// slivers stream from the L2-resident slab. Capped at 512: on a large L1
    /// a longer `kc` only shrinks the other two blocks, measured here as a
    /// loss at n = 256..512 and no gain above.
    static constexpr idx kc =
        std::clamp<idx>(detail::round_down((NUM_K_L1_BYTES / 2) / (sizeof(T) * nr), 8), 8, 512);
    /// Rows of the packed A slab: half of L2, at most 1 MiB.
    static constexpr idx mc = std::max<idx>(
        mr,
        detail::round_down(std::min<idx>(NUM_K_L2_BYTES / 2, idx{1} << 20) / (sizeof(T) * kc), mr));
    /// Columns of the packed B panel.
    static constexpr idx nc =
        std::max<idx>(nr, detail::round_down(NUM_K_GEMM_PANEL_BYTES / (sizeof(T) * kc), nr));
    /// Elements of workspace the packed panels need at their largest.
    static constexpr idx workspace = (mc * kc) + (kc * nc);
};

/// @brief Workspace elements `gemm` needs for a product of the given shape.
///
/// Never more than `gemm_config<T>::workspace`; smaller products need less.
template <std::floating_point T>
[[nodiscard]] constexpr idx gemm_workspace(idx m, idx n, idx k) noexcept {
    using cfg = gemm_config<T>;
    const idx kb = std::min(k, cfg::kc);
    const idx mb = detail::round_up(std::min(m, cfg::mc), cfg::mr);
    const idx nb = detail::round_up(std::min(n, cfg::nc), cfg::nr);
    return (mb * kb) + (kb * nb);
}

namespace detail {

/// @brief Pack `rows x depth` of A, scaled by alpha, as `mr`-row tiles.
///
/// Element (i, p) of the operand is `A[i*row_stride + p*col_stride]`, so a
/// transposed operand is the same call with the strides swapped. Tile `t`
/// occupies `[t*mr*depth, (t+1)*mr*depth)`, laid out `depth` groups of `mr`
/// consecutive rows; rows past `rows` in the last tile are zero.
template <std::floating_point T>
inline void gemm_pack_a(T *NUM_K_RESTRICT dst, const T *NUM_K_RESTRICT A, idx row_stride,
                        idx col_stride, T alpha, idx rows, idx depth) noexcept {
    constexpr idx mr = gemm_config<T>::mr;
    for (idx i0 = 0; i0 < rows; i0 += mr) {
        const idx valid = std::min(mr, rows - i0);
        if (valid == mr && row_stride == 1) {
            // Transposed operand: each of the tile's rows is contiguous in
            // memory, so sweep `p` along it and scatter into the tile.
            for (idx a = 0; a < mr; ++a) {
                const T *NUM_K_RESTRICT row = A + i0 + a;
                for (idx p = 0; p < depth; ++p) {
                    dst[(p * mr) + a] = alpha * row[p * col_stride];
                }
            }
            dst += mr * depth;
        } else if (valid == mr) {
            for (idx p = 0; p < depth; ++p) {
                const T *NUM_K_RESTRICT column = A + (i0 * row_stride) + (p * col_stride);
                for (idx a = 0; a < mr; ++a) {
                    dst[a] = alpha * column[a * row_stride];
                }
                dst += mr;
            }
        } else {
            for (idx p = 0; p < depth; ++p) {
                const T *NUM_K_RESTRICT column = A + (i0 * row_stride) + (p * col_stride);
                for (idx a = 0; a < valid; ++a) {
                    dst[a] = alpha * column[a * row_stride];
                }
                for (idx a = valid; a < mr; ++a) {
                    dst[a] = T(0);
                }
                dst += mr;
            }
        }
    }
}

/// @brief Pack `depth x cols` of B as `nr`-column tiles.
///
/// Element (p, j) of the operand is `B[p*row_stride + j*col_stride]`. Tile `t`
/// occupies `[t*nr*depth, (t+1)*nr*depth)`, laid out `depth` groups of `nr`
/// consecutive columns; columns past `cols` in the last tile are zero.
template <std::floating_point T>
inline void gemm_pack_b(T *NUM_K_RESTRICT dst, const T *NUM_K_RESTRICT B, idx row_stride,
                        idx col_stride, idx depth, idx cols) noexcept {
    constexpr idx nr = gemm_config<T>::nr;
    for (idx j0 = 0; j0 < cols; j0 += nr) {
        const idx valid = std::min(nr, cols - j0);
        if (valid == nr && col_stride == 1) {
            for (idx p = 0; p < depth; ++p) {
                const T *NUM_K_RESTRICT row = B + (p * row_stride) + j0;
                NUM_K_IVDEP
                for (idx b = 0; b < nr; ++b) {
                    dst[b] = row[b];
                }
                dst += nr;
            }
        } else if (row_stride == 1) {
            // Transposed operand: each tile column is contiguous, so sweep
            // `p` along it; columns past `cols` are zero.
            for (idx b = 0; b < nr; ++b) {
                if (b < valid) {
                    const T *NUM_K_RESTRICT column = B + ((j0 + b) * col_stride);
                    for (idx p = 0; p < depth; ++p) {
                        dst[(p * nr) + b] = column[p];
                    }
                } else {
                    for (idx p = 0; p < depth; ++p) {
                        dst[(p * nr) + b] = T(0);
                    }
                }
            }
            dst += nr * depth;
        } else {
            for (idx p = 0; p < depth; ++p) {
                const T *NUM_K_RESTRICT row = B + (p * row_stride) + (j0 * col_stride);
                for (idx b = 0; b < valid; ++b) {
                    dst[b] = row[b * col_stride];
                }
                for (idx b = valid; b < nr; ++b) {
                    dst[b] = T(0);
                }
                dst += nr;
            }
        }
    }
}

/// @brief `C(0:rows, 0:cols) += Ap * Bp` for one packed `mr x nr` tile.
///
/// `Ap` is one A tile from `gemm_pack_a` and `Bp` one B tile from
/// `gemm_pack_b`, both of inner length `depth`. `rows` and `cols` are the
/// valid extent when the tile overhangs C; the padded lanes are computed and
/// discarded, so the microkernel never sees an edge.
template <std::floating_point T>
NUM_K_AINLINE void gemm_micro(T *NUM_K_RESTRICT C, idx ldc, const T *NUM_K_RESTRICT Ap,
                              const T *NUM_K_RESTRICT Bp, idx depth, idx rows, idx cols) noexcept {
    using cfg = gemm_config<T>;
    constexpr idx mr = cfg::mr;
    constexpr idx nr = cfg::nr;
#if defined(NUM_K_VECTOR_EXT)
    using vector = typename vector_of<T, NUM_K_VECTOR_BYTES>::type;
    constexpr idx width = cfg::width;
    constexpr idx nv = cfg::b_vectors;

    vector acc[mr][nv]{};
    for (idx p = 0; p < depth; ++p) {
        vector b[nv];
        for (idx v = 0; v < nv; ++v) {
            std::memcpy(&b[v], Bp + (p * nr) + (v * width), sizeof(vector));
        }
        const T *NUM_K_RESTRICT a = Ap + (p * mr);
        for (idx r = 0; r < mr; ++r) {
            vector broadcast;
            for (idx k = 0; k < width; ++k) {
                broadcast[k] = a[r];
            }
            for (idx v = 0; v < nv; ++v) {
                acc[r][v] += broadcast * b[v];
            }
        }
    }

    if (rows == mr && cols == nr) {
        for (idx r = 0; r < mr; ++r) {
            T *NUM_K_RESTRICT c = C + (r * ldc);
            for (idx v = 0; v < nv; ++v) {
                vector current;
                std::memcpy(&current, c + (v * width), sizeof(vector));
                current += acc[r][v];
                std::memcpy(c + (v * width), &current, sizeof(vector));
            }
        }
        return;
    }
    for (idx r = 0; r < rows; ++r) {
        T *NUM_K_RESTRICT c = C + (r * ldc);
        for (idx j = 0; j < cols; ++j) {
            c[j] += acc[r][j / width][j % width];
        }
    }
#else
    T acc[mr][nr]{};
    for (idx p = 0; p < depth; ++p) {
        const T *NUM_K_RESTRICT b = Bp + (p * nr);
        const T *NUM_K_RESTRICT a = Ap + (p * mr);
        for (idx r = 0; r < mr; ++r) {
            NUM_K_IVDEP
            for (idx j = 0; j < nr; ++j) {
                acc[r][j] += a[r] * b[j];
            }
        }
    }
    for (idx r = 0; r < rows; ++r) {
        T *NUM_K_RESTRICT c = C + (r * ldc);
        for (idx j = 0; j < cols; ++j) {
            c[j] += acc[r][j];
        }
    }
#endif
}

/// @brief Scale C by beta in place, `m x n` with row stride `ldc`.
template <std::floating_point T>
inline void gemm_scale_c(T *NUM_K_RESTRICT C, idx ldc, T beta, idx m, idx n) noexcept {
    if (beta == T(1)) {
        return;
    }
    for (idx i = 0; i < m; ++i) {
        if (beta == T(0)) {
            fill(C + (i * ldc), T(0), n);
        } else {
            scale(C + (i * ldc), beta, n);
        }
    }
}

/// @brief The blocked product over strided operands, into caller workspace.
///
/// Element (i, p) of A is `A[i*a_rows + p*a_cols]` and element (p, j) of B is
/// `B[p*b_rows + j*b_cols]`, so every transposition variant is this routine
/// with the strides swapped. `work` holds at least `gemm_workspace<T>(m, n, k)`
/// elements.
template <std::floating_point T>
inline void gemm_strided(T *NUM_K_RESTRICT C, idx ldc, const T *NUM_K_RESTRICT A, idx a_rows,
                         idx a_cols, const T *NUM_K_RESTRICT B, idx b_rows, idx b_cols, T alpha,
                         T beta, idx m, idx n, idx k, T *NUM_K_RESTRICT work) noexcept {
    using cfg = gemm_config<T>;
    gemm_scale_c(C, ldc, beta, m, n);
    if (m == 0 || n == 0 || k == 0 || alpha == T(0)) {
        return;
    }

    const idx kc = std::min(k, cfg::kc);
    T *NUM_K_RESTRICT Ap = work;
    T *NUM_K_RESTRICT Bp = work + (round_up(std::min(m, cfg::mc), cfg::mr) * kc);

    for (idx jc = 0; jc < n; jc += cfg::nc) {
        const idx nb = std::min(cfg::nc, n - jc);
        for (idx pc = 0; pc < k; pc += cfg::kc) {
            const idx kb = std::min(cfg::kc, k - pc);
            gemm_pack_b(Bp, B + (pc * b_rows) + (jc * b_cols), b_rows, b_cols, kb, nb);
            for (idx ic = 0; ic < m; ic += cfg::mc) {
                const idx mb = std::min(cfg::mc, m - ic);
                gemm_pack_a(Ap, A + (ic * a_rows) + (pc * a_cols), a_rows, a_cols, alpha, mb, kb);
                for (idx jr = 0; jr < nb; jr += cfg::nr) {
                    const idx cols = std::min(cfg::nr, nb - jr);
                    for (idx ir = 0; ir < mb; ir += cfg::mr) {
                        const idx rows = std::min(cfg::mr, mb - ir);
                        gemm_micro(C + ((ic + ir) * ldc) + jc + jr, ldc, Ap + (ir * kb),
                                   Bp + (jr * kb), kb, rows, cols);
                    }
                }
            }
        }
    }
}

/// @brief Per-thread packing storage for the overloads that take no workspace.
///
/// The one place in the kernel with storage of its own. It is static and
/// thread-local rather than heap-allocated, so the routine still never calls
/// an allocator and is still safe to call from any thread; the cost is
/// `gemm_config<T>::workspace` elements of TLS per thread that uses it.
/// @brief Packing storage for the overloads that take no workspace.
///
/// A per-thread static buffer of `gemm_config<T>::workspace` elements: the
/// one place in the kernel with storage of its own. It never calls an
/// allocator and is safe to call from any thread; the cost is that many
/// elements of TLS per thread that uses it. `get()` is the buffer.
///
/// One toolchain is the exception. GCC on macOS implements `thread_local`
/// through emulated TLS, whose control block is a weak symbol in `.data`, and
/// its Darwin backend then emits other local data in that section (among
/// them every `std::source_location` record) as an offset from that weak
/// symbol. Once the linker coalesces the weak copies, each such offset lands
/// in one translation unit's data, and every caller's source location comes
/// out as the first unit's. No `thread_local` in a header survives that, so
/// on that toolchain alone the workspace is allocated per call and freed
/// when this object goes out of scope.
template <std::floating_point T>
class gemm_scratch {
  public:
#if defined(__APPLE__) && defined(__GNUC__) && !defined(__clang__)
    gemm_scratch() noexcept
        : buffer_(static_cast<T *>(std::malloc(gemm_config<T>::workspace * sizeof(T)))) {}
    ~gemm_scratch() { std::free(buffer_); }
    gemm_scratch(const gemm_scratch &) = delete;
    gemm_scratch &operator=(const gemm_scratch &) = delete;
    [[nodiscard]] T *get() const noexcept { return buffer_; }

  private:
    T *buffer_;
#else
    [[nodiscard]] T *get() const noexcept {
        alignas(64) static thread_local T buffer[gemm_config<T>::workspace];
        return buffer;
    }
#endif
};

} // namespace detail

/// @brief Dense matrix product `C <- alpha*A*B + beta*C`, with row strides and
/// caller-provided workspace.
///
/// A is `m x k`, B is `k x n`, C is `m x n`; `work` holds at least
/// `gemm_workspace<T>(m, n, k)` elements. A, B, C and `work` must not overlap.
/// See @ref gemm_config for the blocking.
template <std::floating_point T>
inline void gemm(T *NUM_K_RESTRICT C, idx ldc, const T *NUM_K_RESTRICT A, idx lda,
                 const T *NUM_K_RESTRICT B, idx ldb, T alpha, T beta, idx m, idx n, idx k,
                 T *NUM_K_RESTRICT work) noexcept {
    detail::gemm_strided(C, ldc, A, lda, idx{1}, B, ldb, idx{1}, alpha, beta, m, n, k, work);
}

/// @brief Dense matrix product `C <- alpha*A*B + beta*C`, with row strides.
///
/// Packs through a per-thread static buffer (see `detail::gemm_scratch`);
/// pass a workspace explicitly to keep the call free of any state.
template <std::floating_point T>
inline void gemm(T *NUM_K_RESTRICT C, idx ldc, const T *NUM_K_RESTRICT A, idx lda,
                 const T *NUM_K_RESTRICT B, idx ldb, T alpha, T beta, idx m, idx n,
                 idx k) noexcept {
    const detail::gemm_scratch<T> scratch;
    gemm(C, ldc, A, lda, B, ldb, alpha, beta, m, n, k, scratch.get());
}

template <std::floating_point T>
inline void gemm(T *NUM_K_RESTRICT C, const T *NUM_K_RESTRICT A, const T *NUM_K_RESTRICT B, T alpha,
                 T beta, idx m, idx n, idx k) noexcept {
    gemm(C, n, A, k, B, n, alpha, beta, m, n, k);
}

// Packed LU without row pivoting. Suitable for matrices whose structure
// guarantees nonzero pivots, including the M-matrices used by ELSE.
/// @brief In-place LU without row pivoting. Returns false if a pivot fell below tolerance.
template <std::floating_point T>
[[nodiscard]] inline bool lu_no_pivot(T *A, idx n) noexcept {
    constexpr T tolerance = T(1e-15);
    bool nonsingular = true;
    for (idx k = 0; k < n; ++k) {
        if (std::abs(A[k * n + k]) < tolerance) {
            nonsingular = false;
            continue;
        }
        const T inverse_pivot = T(1) / A[k * n + k];
        for (idx i = k + 1; i < n; ++i) {
            A[i * n + k] *= inverse_pivot;
            const T multiplier = A[i * n + k];
            for (idx j = k + 1; j < n; ++j)
                A[i * n + j] -= multiplier * A[k * n + j];
        }
    }
    return nonsingular;
}

/// @brief Solves several right-hand sides from an unpivoted LU factor.
template <std::floating_point T>
inline void lu_no_pivot_solve_multiple(T *X, const T *LU, idx n, idx columns) noexcept {
    for (idx i = 0; i < n; ++i)
        for (idx j = 0; j < i; ++j)
            for (idx c = 0; c < columns; ++c)
                X[i * columns + c] -= LU[i * n + j] * X[j * columns + c];
    for (idx i = n; i-- > 0;) {
        for (idx j = i + 1; j < n; ++j)
            for (idx c = 0; c < columns; ++c)
                X[i * columns + c] -= LU[i * n + j] * X[j * columns + c];
        for (idx c = 0; c < columns; ++c)
            X[i * columns + c] /= LU[i * n + i];
    }
}

/// @brief Solves \f$A^T x = b\f$ for several right-hand sides from an unpivoted LU factor.
template <std::floating_point T>
inline void lu_no_pivot_solve_transpose_multiple(T *X, const T *LU, idx n, idx columns) noexcept {
    for (idx i = 0; i < n; ++i) {
        for (idx j = 0; j < i; ++j)
            for (idx c = 0; c < columns; ++c)
                X[i * columns + c] -= LU[j * n + i] * X[j * columns + c];
        for (idx c = 0; c < columns; ++c)
            X[i * columns + c] /= LU[i * n + i];
    }
    for (idx i = n; i-- > 0;)
        for (idx j = i + 1; j < n; ++j)
            for (idx c = 0; c < columns; ++c)
                X[i * columns + c] -= LU[j * n + i] * X[j * columns + c];
}

/// @brief Symmetric rank-k update of the lower triangle, `C <- alpha*A*A^T + beta*C`.
///
/// `A` is `rows x columns`; `C` is a row-major `rows x rows` matrix.  Only
/// `C(i,j)` for `j <= i` is touched, which is the update required by lower
/// Cholesky and avoids doing work for the implied symmetric half.
///
/// Column strips of the triangle: each diagonal block is formed entry by entry
/// so nothing above the diagonal is written, and the rectangle beneath it is
/// one `gemm` with `A^T` as the right operand, so the bulk of the work runs at
/// `gemm` speed.
namespace detail {

/// @brief `syrk_lower` over column strips of width `strip`.
///
/// Each diagonal block is formed entry by entry when `strip` is the narrow
/// base width, else by recursion at the base width; the rectangle beneath it
/// is one `gemm`. Two levels because the base width is what keeps the
/// entry-wise part negligible, while wide strips are what let `gemm` reuse
/// its packed A slab.
template <std::floating_point T>
inline void syrk_lower_strips(T *NUM_K_RESTRICT C, idx ldc, const T *NUM_K_RESTRICT A, idx lda,
                              T alpha, T beta, idx rows, idx columns, idx strip,
                              T *NUM_K_RESTRICT work) noexcept {
    constexpr idx base = 4 * gemm_config<T>::nr;
    for (idx j0 = 0; j0 < rows; j0 += strip) {
        const idx j1 = std::min(rows, j0 + strip);
        if (strip > base) {
            syrk_lower_strips(C + (j0 * ldc) + j0, ldc, A + (j0 * lda), lda, alpha, beta, j1 - j0,
                              columns, base, work);
        } else {
            for (idx i = j0; i < j1; ++i) {
                T *NUM_K_RESTRICT c_row = C + (i * ldc);
                const T *NUM_K_RESTRICT a_i = A + (i * lda);
                for (idx j = j0; j <= i; ++j) {
                    const T *NUM_K_RESTRICT a_j = A + (j * lda);
                    const T sum = reduce<T>(columns, [a_i, a_j](idx p) { return a_i[p] * a_j[p]; });
                    c_row[j] = (alpha * sum) + (beta * c_row[j]);
                }
            }
        }
        if (j1 < rows) {
            gemm_strided(C + (j1 * ldc) + j0, ldc, A + (j1 * lda), lda, idx{1}, A + (j0 * lda),
                         idx{1}, lda, alpha, beta, rows - j1, j1 - j0, columns, work);
        }
    }
}

} // namespace detail

template <std::floating_point T>
inline void syrk_lower(T *NUM_K_RESTRICT C, idx ldc, const T *NUM_K_RESTRICT A, idx lda, T alpha,
                       T beta, idx rows, idx columns) noexcept {
    const detail::gemm_scratch<T> scratch;
    detail::syrk_lower_strips(C, ldc, A, lda, alpha, beta, rows, columns, gemm_config<T>::mc,
                              scratch.get());
}

template <std::floating_point T>
inline void syrk_lower(T *NUM_K_RESTRICT C, const T *NUM_K_RESTRICT A, T alpha, T beta, idx rows,
                       idx columns) noexcept {
    syrk_lower(C, rows, A, columns, alpha, beta, rows, columns);
}

/// @brief Dense product `C <- alpha*A^T*B + beta*C`, with row strides.
///
/// A is `rows x a_cols`, B is `rows x b_cols`, and C is `a_cols x b_cols`.
template <std::floating_point T>
inline void gemm_transpose_left(T *NUM_K_RESTRICT C, idx ldc, const T *NUM_K_RESTRICT A, idx lda,
                                const T *NUM_K_RESTRICT B, idx ldb, T alpha, T beta, idx rows,
                                idx a_cols, idx b_cols) noexcept {
    const detail::gemm_scratch<T> scratch;
    detail::gemm_strided(C, ldc, A, idx{1}, lda, B, ldb, idx{1}, alpha, beta, a_cols, b_cols, rows,
                         scratch.get());
}

/// @brief Block projection coefficients \f$h \leftarrow V^T w\f$.
template <std::floating_point T>
inline void project_columns(T *NUM_K_RESTRICT h, const T *NUM_K_RESTRICT V, idx ldv,
                            const T *NUM_K_RESTRICT w, idx rows, idx columns) noexcept {
    fill(h, T(0), columns);
    for (idx r = 0; r < rows; ++r) {
        const T wr = w[r];
        const T *NUM_K_RESTRICT v_row = V + (r * ldv);
        NUM_K_IVDEP
        for (idx j = 0; j < columns; ++j) {
            h[j] += v_row[j] * wr;
        }
    }
}

/// @brief Block linear combination \f$y \leftarrow \alpha Vc + \beta y\f$.
template <std::floating_point T>
inline void combine_columns(T *NUM_K_RESTRICT y, const T *NUM_K_RESTRICT V, idx ldv,
                            const T *NUM_K_RESTRICT coefficients, T alpha, T beta, idx rows,
                            idx columns) noexcept {
    for (idx r = 0; r < rows; ++r) {
        const T *v_row = V + (r * ldv);
        const T sum = detail::reduce<T>(
            columns, [v_row, coefficients](idx j) { return v_row[j] * coefficients[j]; });
        y[r] = (alpha * sum) + (beta * y[r]);
    }
}

/// @brief Modified Gram--Schmidt against row-major basis columns.
///
/// This intentionally retains sequential projection/update ordering; callers
/// requiring the faster classical block operation use `project_columns` followed
/// by `combine_columns` and accept its different stability contract.
template <std::floating_point T>
inline void mgs_columns(T *NUM_K_RESTRICT v, const T *NUM_K_RESTRICT basis, idx ldb, idx rows,
                        idx columns, T *coefficients = nullptr) noexcept {
    for (idx column = 0; column < columns; ++column) {
        T projection = T(0);
        for (idx row = 0; row < rows; ++row) {
            projection += basis[(row * ldb) + column] * v[row];
        }
        if (coefficients != nullptr) {
            coefficients[column] = projection;
        }
        for (idx row = 0; row < rows; ++row) {
            v[row] -= projection * basis[(row * ldb) + column];
        }
    }
}

/// @brief Out-of-place matrix transpose \f$B = A^T\f$ for \f$A \in \mathbb{R}^{m \times n}\f$.
template <std::floating_point T>
NUM_K_AINLINE void transpose(T *NUM_K_RESTRICT B, const T *NUM_K_RESTRICT A, idx m,
                             idx n) noexcept {
    for (idx i = 0; i < m; ++i) {
        for (idx j = 0; j < n; ++j) {
            B[(j * m) + i] = A[(i * n) + j];
        }
    }
}

/// @brief Rank-1 matrix update \f$A \leftarrow A + \alpha \mathbf{x} \mathbf{y}^T\f$ on an \f$m
/// \times n\f$ block with row stride `lda`.
template <std::floating_point T>
NUM_K_AINLINE void ger(T *NUM_K_RESTRICT A, idx lda, const T *NUM_K_RESTRICT x,
                       const T *NUM_K_RESTRICT y, T alpha, idx m, idx n) noexcept {
    for (idx i = 0; i < m; ++i) {
        T *NUM_K_RESTRICT row = A + (i * lda);
        const T axi = alpha * x[i];
        NUM_K_IVDEP
        for (idx j = 0; j < n; ++j) {
            row[j] += axi * y[j];
        }
    }
}

/// @brief Rank-1 matrix update \f$A \leftarrow A + \alpha \mathbf{x} \mathbf{y}^T\f$ for \f$A \in
/// \mathbb{R}^{m \times n}\f$.
template <std::floating_point T>
NUM_K_AINLINE void ger(T *NUM_K_RESTRICT A, const T *NUM_K_RESTRICT x, const T *NUM_K_RESTRICT y,
                       T alpha, idx m, idx n) noexcept {
    ger(A, n, x, y, alpha, m, n);
}

/// @brief Forward substitution solving lower triangular system \f$L \mathbf{x} = \mathbf{b}\f$
/// (\f$L \in \mathbb{R}^{n \times n}\f$).
template <std::floating_point T>
NUM_K_AINLINE void trsv_lower(T *NUM_K_RESTRICT x, const T *NUM_K_RESTRICT L,
                              const T *NUM_K_RESTRICT b, idx n) noexcept {
    for (idx i = 0; i < n; ++i) {
        T s = b[i];
        const T *row = L + (i * n);
        for (idx j = 0; j < i; ++j) {
            s -= row[j] * x[j];
        }
        x[i] = s / row[i];
    }
}

/// @brief In-place lower triangular solve.  This is the alias-safe form of
/// `trsv_lower(x, L, x, n)` and therefore carries no contradictory no-alias
/// promise.
template <std::floating_point T>
NUM_K_AINLINE void trsv_lower_inplace(T *x, const T *NUM_K_RESTRICT L, idx n) noexcept {
    for (idx i = 0; i < n; ++i) {
        T s = x[i];
        const T *row = L + (i * n);
        for (idx j = 0; j < i; ++j) {
            s -= row[j] * x[j];
        }
        x[i] = s / row[i];
    }
}

/// @brief Explicit alias-safe lower triangular solve; `x` and `b` may coincide.
template <std::floating_point T>
NUM_K_AINLINE void trsv_lower(contract::alias_safe_t, T *x, const T *NUM_K_RESTRICT L, const T *b,
                              idx n) noexcept {
    if (x != b) {
        NUM_K_IVDEP
        for (idx i = 0; i < n; ++i) {
            x[i] = b[i];
        }
    }
    trsv_lower_inplace(x, L, n);
}

namespace detail {

/// @brief Rows or columns of the triangle solved per unblocked step in the
/// `trsm` routines. Everything outside the diagonal block is a `gemm`.
inline constexpr idx trsm_block = 64;

/// @brief Unblocked `L*X=B` on one diagonal block.
///
/// Element (i, j) of the triangle is `L[i*row_stride + j*col_stride]`, so a
/// transposed upper factor is the same call with the strides swapped; `unit`
/// takes the diagonal as one without reading it.
template <std::floating_point T>
inline void trsm_lower_block(T *NUM_K_RESTRICT X, idx ldx, const T *NUM_K_RESTRICT L,
                             idx row_stride, idx col_stride, idx n, idx nrhs, bool unit) noexcept {
    for (idx i = 0; i < n; ++i) {
        T *NUM_K_RESTRICT x_row = X + (i * ldx);
        for (idx j = 0; j < i; ++j) {
            const T lij = L[(i * row_stride) + (j * col_stride)];
            const T *NUM_K_RESTRICT solved_row = X + (j * ldx);
            NUM_K_IVDEP
            for (idx r = 0; r < nrhs; ++r) {
                x_row[r] -= lij * solved_row[r];
            }
        }
        if (!unit) {
            const T inv_diag = T(1) / L[(i * row_stride) + (i * col_stride)];
            NUM_K_IVDEP
            for (idx r = 0; r < nrhs; ++r) {
                x_row[r] *= inv_diag;
            }
        }
    }
}

/// @brief Unblocked `L^T*X=B` on one diagonal block, strides as for `trsm_lower_block`.
template <std::floating_point T>
inline void trsm_lower_transpose_block(T *NUM_K_RESTRICT X, idx ldx, const T *NUM_K_RESTRICT L,
                                       idx row_stride, idx col_stride, idx n, idx nrhs,
                                       bool unit) noexcept {
    for (idx i = n; i-- > 0;) {
        T *NUM_K_RESTRICT row = X + (i * ldx);
        for (idx k = i + 1; k < n; ++k) {
            const T lki = L[(k * row_stride) + (i * col_stride)];
            const T *NUM_K_RESTRICT solved = X + (k * ldx);
            NUM_K_IVDEP
            for (idx r = 0; r < nrhs; ++r) {
                row[r] -= lki * solved[r];
            }
        }
        if (!unit) {
            const T inv = T(1) / L[(i * row_stride) + (i * col_stride)];
            NUM_K_IVDEP
            for (idx r = 0; r < nrhs; ++r) {
                row[r] *= inv;
            }
        }
    }
}

/// @brief Blocked `L*X=B`: solve a diagonal block, then subtract its
/// contribution from every row below with one `gemm`.
template <std::floating_point T>
inline void trsm_lower_blocked(T *NUM_K_RESTRICT X, idx ldx, const T *NUM_K_RESTRICT L,
                               idx row_stride, idx col_stride, idx n, idx nrhs,
                               bool unit) noexcept {
    const gemm_scratch<T> scratch;
    T *NUM_K_RESTRICT work = scratch.get();
    for (idx i0 = 0; i0 < n; i0 += trsm_block) {
        const idx i1 = std::min(n, i0 + trsm_block);
        trsm_lower_block(X + (i0 * ldx), ldx, L + (i0 * row_stride) + (i0 * col_stride), row_stride,
                         col_stride, i1 - i0, nrhs, unit);
        if (i1 < n) {
            gemm_strided(X + (i1 * ldx), ldx, L + (i1 * row_stride) + (i0 * col_stride), row_stride,
                         col_stride, X + (i0 * ldx), ldx, idx{1}, T(-1), T(1), n - i1, nrhs,
                         i1 - i0, work);
        }
    }
}

/// @brief Blocked `L^T*X=B`, from the last diagonal block upward; the rows
/// above each block are updated with `L^T` expressed as a stride swap.
template <std::floating_point T>
inline void trsm_lower_transpose_blocked(T *NUM_K_RESTRICT X, idx ldx, const T *NUM_K_RESTRICT L,
                                         idx row_stride, idx col_stride, idx n, idx nrhs,
                                         bool unit) noexcept {
    const gemm_scratch<T> scratch;
    T *NUM_K_RESTRICT work = scratch.get();
    idx i1 = n;
    while (i1 > 0) {
        const idx i0 = i1 > trsm_block ? i1 - trsm_block : 0;
        trsm_lower_transpose_block(X + (i0 * ldx), ldx, L + (i0 * row_stride) + (i0 * col_stride),
                                   row_stride, col_stride, i1 - i0, nrhs, unit);
        if (i0 > 0) {
            gemm_strided(X, ldx, L + (i0 * row_stride), col_stride, row_stride, X + (i0 * ldx), ldx,
                         idx{1}, T(-1), T(1), i0, nrhs, i1 - i0, work);
        }
        i1 = i0;
    }
}

/// @brief Rows solved together by `trsm_lower_transpose_right_block`.
inline constexpr idx trsm_row_batch = 16;

/// @brief Unblocked `X*L^T=B` on one diagonal block of at most `trsm_block` columns.
///
/// Each row's solve is a chain of dependent updates, so a row at a time runs
/// scalar. Instead a batch of rows is copied into a column-major tile on the
/// stack, where the update of column `j` by column `k` is a contiguous axpy
/// across the batch, and copied back once solved.
template <std::floating_point T>
inline void trsm_lower_transpose_right_block(T *NUM_K_RESTRICT X, idx ldx,
                                             const T *NUM_K_RESTRICT L, idx ldl, idx rows,
                                             idx n) noexcept {
    constexpr idx batch = trsm_row_batch;
    T tile[trsm_block][batch];
    for (idx r0 = 0; r0 < rows; r0 += batch) {
        const idx count = std::min(batch, rows - r0);
        for (idx r = 0; r < count; ++r) {
            const T *NUM_K_RESTRICT row = X + ((r0 + r) * ldx);
            for (idx j = 0; j < n; ++j) {
                tile[j][r] = row[j];
            }
        }
        for (idx j = 0; j < n; ++j) {
            // Column j accumulates in registers across the k loop; the tile
            // row it came from is only touched at the ends.
            T acc[batch];
            for (idx r = 0; r < batch; ++r) {
                acc[r] = tile[j][r];
            }
            const T *NUM_K_RESTRICT l_row = L + (j * ldl);
            for (idx k = 0; k < j; ++k) {
                const T ljk = l_row[k];
                NUM_K_IVDEP
                for (idx r = 0; r < batch; ++r) {
                    acc[r] -= ljk * tile[k][r];
                }
            }
            const T inv = T(1) / l_row[j];
            for (idx r = 0; r < batch; ++r) {
                tile[j][r] = acc[r] * inv;
            }
        }
        for (idx r = 0; r < count; ++r) {
            T *NUM_K_RESTRICT row = X + ((r0 + r) * ldx);
            for (idx j = 0; j < n; ++j) {
                row[j] = tile[j][r];
            }
        }
    }
}

} // namespace detail

/// @brief Solves `L*X=B` in-place for row-major `X` with `nrhs` columns.
///
/// Blocked by `detail::trsm_block` rows of the triangle: each diagonal block
/// is a short substitution, and the rows beneath it are updated by `gemm`, so
/// for `n` past one block the work runs at `gemm` speed. Vectorization within
/// a block is across the independent right-hand sides.
template <std::floating_point T>
inline void trsm_lower_inplace(T *NUM_K_RESTRICT X, idx ldx, const T *NUM_K_RESTRICT L, idx n,
                               idx nrhs) noexcept {
    detail::trsm_lower_blocked(X, ldx, L, n, idx{1}, n, nrhs, false);
}

/// @brief Solves `L*X=B` in-place when `L` is unit lower triangular.
template <std::floating_point T>
inline void trsm_unit_lower_inplace(T *NUM_K_RESTRICT X, idx ldx, const T *NUM_K_RESTRICT L,
                                    idx ldl, idx n, idx nrhs) noexcept {
    detail::trsm_lower_blocked(X, ldx, L, ldl, idx{1}, n, nrhs, true);
}

/// @brief Solves `L^T*X=B` in-place for row-major multiple right-hand sides.
template <std::floating_point T>
inline void trsm_lower_transpose_inplace(T *NUM_K_RESTRICT X, idx ldx, const T *NUM_K_RESTRICT L,
                                         idx n, idx nrhs) noexcept {
    detail::trsm_lower_transpose_blocked(X, ldx, L, n, idx{1}, n, nrhs, false);
}

/// @brief Solves `L^T*X=B` in-place when `L` is unit lower triangular.
template <std::floating_point T>
inline void trsm_unit_lower_transpose_inplace(T *NUM_K_RESTRICT X, idx ldx,
                                              const T *NUM_K_RESTRICT L, idx ldl, idx n,
                                              idx nrhs) noexcept {
    detail::trsm_lower_transpose_blocked(X, ldx, L, ldl, idx{1}, n, nrhs, true);
}

/// @brief Solves `U*X=B` in-place for upper triangular `U` with row stride `ldu`.
///
/// `U` is the transpose of a lower factor, so this is the transposed-lower
/// solve with the strides swapped.
template <std::floating_point T>
inline void trsm_upper_inplace(T *NUM_K_RESTRICT X, idx ldx, const T *NUM_K_RESTRICT U, idx ldu,
                               idx n, idx nrhs, bool unit = false) noexcept {
    detail::trsm_lower_transpose_blocked(X, ldx, U, idx{1}, ldu, n, nrhs, unit);
}

/// @brief Solves `U^T*X=B` in-place for upper triangular `U` with row stride `ldu`.
template <std::floating_point T>
inline void trsm_upper_transpose_inplace(T *NUM_K_RESTRICT X, idx ldx, const T *NUM_K_RESTRICT U,
                                         idx ldu, idx n, idx nrhs, bool unit = false) noexcept {
    detail::trsm_lower_blocked(X, ldx, U, idx{1}, ldu, n, nrhs, unit);
}

/// @brief Solves `X*L^T=B` in-place for row-major panel rows.
///
/// `X` is `rows x n`, `L` is lower triangular `n x n`; each row is an
/// independent right-side solve. This is the panel solve in blocked
/// Cholesky: `L21 <- A21 * L11^{-T}`. Blocked by columns of `X`: after a
/// diagonal block is solved, the columns to its right are updated by `gemm`.
template <std::floating_point T>
inline void trsm_lower_transpose_right_inplace(T *NUM_K_RESTRICT X, idx ldx,
                                               const T *NUM_K_RESTRICT L, idx ldl, idx rows,
                                               idx n) noexcept {
    const detail::gemm_scratch<T> scratch;
    T *NUM_K_RESTRICT work = scratch.get();
    for (idx j0 = 0; j0 < n; j0 += detail::trsm_block) {
        const idx j1 = std::min(n, j0 + detail::trsm_block);
        detail::trsm_lower_transpose_right_block(X + j0, ldx, L + (j0 * ldl) + j0, ldl, rows,
                                                 j1 - j0);
        if (j1 < n) {
            detail::gemm_strided(X + j1, ldx, X + j0, ldx, idx{1}, L + (j1 * ldl) + j0, idx{1}, ldl,
                                 T(-1), T(1), rows, n - j1, j1 - j0, work);
        }
    }
}

/// @brief Back substitution solving upper triangular system \f$U \mathbf{x} = \mathbf{b}\f$ (\f$U
/// \in \mathbb{R}^{n \times n}\f$).
template <std::floating_point T>
NUM_K_AINLINE void trsv_upper(T *NUM_K_RESTRICT x, const T *NUM_K_RESTRICT U,
                              const T *NUM_K_RESTRICT b, idx n) noexcept {
    for (idx i = n; i-- > 0;) {
        T s = b[i];
        const T *row = U + (i * n);
        for (idx j = i + 1; j < n; ++j) {
            s -= row[j] * x[j];
        }
        x[i] = s / row[i];
    }
}

/// @brief In-place upper triangular solve.
template <std::floating_point T>
NUM_K_AINLINE void trsv_upper_inplace(T *x, const T *NUM_K_RESTRICT U, idx n) noexcept {
    for (idx i = n; i-- > 0;) {
        T s = x[i];
        const T *row = U + (i * n);
        for (idx j = i + 1; j < n; ++j) {
            s -= row[j] * x[j];
        }
        x[i] = s / row[i];
    }
}

/// @brief Explicit alias-safe upper triangular solve; `x` and `b` may coincide.
template <std::floating_point T>
NUM_K_AINLINE void trsv_upper(contract::alias_safe_t, T *x, const T *NUM_K_RESTRICT U, const T *b,
                              idx n) noexcept {
    if (x != b) {
        NUM_K_IVDEP
        for (idx i = 0; i < n; ++i) {
            x[i] = b[i];
        }
    }
    trsv_upper_inplace(x, U, n);
}

// Row Swaps, Transposed Triangular Solves & banded Kernels

/// @brief Swaps rows \f$r_1 \leftrightarrow r_2\f$ of length \f$n\f$ in row-major matrix \f$A\f$
/// with stride `lda`.
template <typename T>
NUM_K_AINLINE void swap_rows(T *NUM_K_RESTRICT A, idx lda, idx r1, idx r2, idx n) noexcept {
    if (r1 != r2) {
        swap(A + (r1 * lda), A + (r2 * lda), n);
    }
}

/// @brief Solves transposed lower triangular system \f$L^T \mathbf{x} = \mathbf{b}\f$ (or in-place
/// \f$\mathbf{x} \leftarrow L^{-T} \mathbf{x}\f$).
template <std::floating_point T>
NUM_K_AINLINE void trsv_transpose_lower(T *NUM_K_RESTRICT x, const T *NUM_K_RESTRICT L, idx lda,
                                        idx n) noexcept {
    for (idx i = n; i-- > 0;) {
        T s = x[i];
        for (idx k = i + 1; k < n; ++k) {
            s -= L[(k * lda) + i] * x[k];
        }
        x[i] = s / L[(i * lda) + i];
    }
}

/// @brief Solves transposed upper triangular system \f$U^T \mathbf{x} = \mathbf{b}\f$ (or in-place
/// \f$\mathbf{x} \leftarrow U^{-T} \mathbf{x}\f$).
template <std::floating_point T>
NUM_K_AINLINE void trsv_transpose_upper(T *NUM_K_RESTRICT x, const T *NUM_K_RESTRICT U, idx lda,
                                        idx n) noexcept {
    for (idx i = 0; i < n; ++i) {
        T s = x[i];
        for (idx k = 0; k < i; ++k) {
            s -= U[(k * lda) + i] * x[k];
        }
        x[i] = s / U[(i * lda) + i];
    }
}

/// @brief Computes banded matrix-vector multiplication \f$\mathbf{y} \leftarrow \alpha A \mathbf{x}
/// + \beta \mathbf{y}\f$ in LAPACK band storage (BLAS GBMV).
template <std::floating_point T>
NUM_K_AINLINE void gbmv(T *NUM_K_RESTRICT y, T alpha, const T *NUM_K_RESTRICT ab, idx ldab, idx kl,
                        idx ku, const T *NUM_K_RESTRICT x, T beta, idx n) noexcept {
    if (beta == T(0)) {
        NUM_K_IVDEP
        for (idx i = 0; i < n; ++i) {
            y[i] = T(0);
        }
    } else if (beta != T(1)) {
        NUM_K_IVDEP
        for (idx i = 0; i < n; ++i) {
            y[i] *= beta;
        }
    }
    const idx kv = ku + kl;
    for (idx j = 0; j < n; ++j) {
        if (x[j] != T(0)) {
            const T temp = alpha * x[j];
            const idx i_start = (j > ku) ? j - ku : 0;
            const idx i_end = std::min(j + kl, n - 1);
            NUM_K_IVDEP
            for (idx i = i_start; i <= i_end; ++i) {
                y[i] += ab[kv + i - j + (j * ldab)] * temp;
            }
        }
    }
}

} // namespace num::kernel
