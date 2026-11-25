/// @file kernel/dense.hpp
/// @brief Dense kernels over raw pointers: matrix products, triangular solves, banded products.
///
/// SPDX-License-Identifier: MIT
/// Part of numerics, (c) 2026 Aditya Dendukuri.
/// https://github.com/AdityaDendukuri/numerics
///
/// Buffers are row-major.
#pragma once

#include "kernel/vector.hpp"
#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstdlib>
#include <cstring>
#include <type_traits>

namespace num::kernel {

/// @brief Matrix-vector product (GEMV): \f$y \leftarrow A x\f$.
template <std::floating_point T>
NUM_K_AINLINE void matvec(T *NUM_K_RESTRICT y, const T *NUM_K_RESTRICT A, const T *NUM_K_RESTRICT x,
                          idx m, idx n) noexcept {
    for (idx i = 0; i < m; ++i) {
        const T *row = A + (i * n);
        y[i] = detail::reduce<T>(n, [row, x](idx j) { return row[j] * x[j]; });
    }
}

/// @brief Transposed matrix-vector product: \f$y \leftarrow A^T x\f$.
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

namespace detail {

[[nodiscard]] constexpr idx round_down(idx value, idx multiple) noexcept {
    return multiple == 0 ? value : (value / multiple) * multiple;
}
[[nodiscard]] constexpr idx round_up(idx value, idx multiple) noexcept {
    return multiple == 0 ? value : ((value + multiple - 1) / multiple) * multiple;
}

} // namespace detail

/// @brief GEMM blocking for this target: an `mr x nr` microkernel over packed panels.
template <std::floating_point T>
struct gemm_config {
#if defined(NUM_K_VECTOR_EXT)
    /// Elements per vector register.
    static constexpr idx width = NUM_K_VECTOR_BYTES / sizeof(T);
    /// B vectors per microkernel.
    static constexpr idx b_vectors = (NUM_K_VECTOR_REGISTERS >= 32 && width <= 2) ? 3 : 2;
    /// Register tile rows.
    static constexpr idx mr =
        std::min<idx>(14, ((NUM_K_VECTOR_REGISTERS - b_vectors - 2) / b_vectors) & ~idx{1});
#else
    static constexpr idx width = 4;
    static constexpr idx b_vectors = 2;
    static constexpr idx mr = 4;
#endif
    /// Register tile columns.
    static constexpr idx nr = b_vectors * width;
    /// Inner block depth; the `kc x nr` B sliver fills half of L1.
    static constexpr idx kc =
        std::clamp<idx>(detail::round_down((NUM_K_L1_BYTES / 2) / (sizeof(T) * nr), 8), 8, 512);
    /// Packed A rows; the slab fills half of L2.
    static constexpr idx mc = std::max<idx>(
        mr,
        detail::round_down(std::min<idx>(NUM_K_L2_BYTES / 2, idx{1} << 20) / (sizeof(T) * kc), mr));
    /// Packed B columns.
    static constexpr idx nc =
        std::max<idx>(nr, detail::round_down(NUM_K_GEMM_PANEL_BYTES / (sizeof(T) * kc), nr));
    /// Workspace elements at the largest block sizes.
    static constexpr idx workspace = (mc * kc) + (kc * nc);
};

/// @brief GEMM workspace in elements for an `m x n x k` product.
template <std::floating_point T>
[[nodiscard]] constexpr idx gemm_workspace(idx m, idx n, idx k) noexcept {
    using cfg = gemm_config<T>;
    const idx kb = std::min(k, cfg::kc);
    const idx mb = detail::round_up(std::min(m, cfg::mc), cfg::mr);
    const idx nb = detail::round_up(std::min(n, cfg::nc), cfg::nr);
    return (mb * kb) + (kb * nb);
}

namespace detail {

/// @brief GEMM A packing into `mr`-row tiles, scaled: \f$A_p \leftarrow \alpha A\f$.
/// Element (i, p) is `A[i*row_stride + p*col_stride]`; padding rows are zero.
template <std::floating_point T>
inline void gemm_pack_a(T *NUM_K_RESTRICT dst, const T *NUM_K_RESTRICT A, idx row_stride,
                        idx col_stride, T alpha, idx rows, idx depth) noexcept {
    constexpr idx mr = gemm_config<T>::mr;
    for (idx i0 = 0; i0 < rows; i0 += mr) {
        const idx valid = std::min(mr, rows - i0);
        if (valid == mr && row_stride == 1) {
            // Transposed operand: sweep each contiguous row into the tile.
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

/// @brief GEMM B packing into `nr`-column tiles: \f$B_p \leftarrow B\f$.
/// Element (p, j) is `B[p*row_stride + j*col_stride]`; padding columns are zero.
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
            // Transposed operand: sweep each contiguous column into the tile.
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

/// @brief GEMM microkernel: \f$C \leftarrow C + A_p B_p\f$ on one packed `mr x nr` tile.
/// Only the leading `rows x cols` of C are written.
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

/// @brief Matrix scaling: \f$C \leftarrow \beta C\f$.
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

/// @brief Strided GEMM: \f$C \leftarrow \alpha A B + \beta C\f$.
/// Swapping a pair of strides transposes that operand; `work` holds `gemm_workspace<T>(m, n, k)`.
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

/// @brief Per-thread GEMM workspace for the overloads that take none.
template <std::floating_point T>
class gemm_scratch {
  public:
    // GCC on macOS emulates thread_local unreliably, so allocate per call there.
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

/// @brief General matrix-matrix product (GEMM): \f$C \leftarrow \alpha A B + \beta C\f$.
/// A is `m x k`, B is `k x n`; `work` holds `gemm_workspace<T>(m, n, k)`. Nothing may overlap.
/// @note Each entry sums over k in ascending order, bit-identical to the naive triple loop.
template <std::floating_point T>
inline void gemm(T *NUM_K_RESTRICT C, idx ldc, const T *NUM_K_RESTRICT A, idx lda,
                 const T *NUM_K_RESTRICT B, idx ldb, T alpha, T beta, idx m, idx n, idx k,
                 T *NUM_K_RESTRICT work) noexcept {
    detail::gemm_strided(C, ldc, A, lda, idx{1}, B, ldb, idx{1}, alpha, beta, m, n, k, work);
}

/// @brief General matrix-matrix product (GEMM): \f$C \leftarrow \alpha A B + \beta C\f$.
/// Uses a per-thread workspace.
template <std::floating_point T>
inline void gemm(T *NUM_K_RESTRICT C, idx ldc, const T *NUM_K_RESTRICT A, idx lda,
                 const T *NUM_K_RESTRICT B, idx ldb, T alpha, T beta, idx m, idx n,
                 idx k) noexcept {
    const detail::gemm_scratch<T> scratch;
    gemm(C, ldc, A, lda, B, ldb, alpha, beta, m, n, k, scratch.get());
}

/// @brief Contiguous GEMM: \f$C \leftarrow \alpha A B + \beta C\f$.
template <std::floating_point T>
inline void gemm(T *NUM_K_RESTRICT C, const T *NUM_K_RESTRICT A, const T *NUM_K_RESTRICT B, T alpha,
                 T beta, idx m, idx n, idx k) noexcept {
    gemm(C, n, A, k, B, n, alpha, beta, m, n, k);
}

namespace detail {

/// @brief Recursive strip step of `syrk_lower`.
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

/// @brief Symmetric rank-k update (SYRK), lower: \f$C \leftarrow \alpha A A^T + \beta C\f$.
/// Only entries with \f$j \le i\f$ are written.
template <std::floating_point T>
inline void syrk_lower(T *NUM_K_RESTRICT C, idx ldc, const T *NUM_K_RESTRICT A, idx lda, T alpha,
                       T beta, idx rows, idx columns) noexcept {
    const detail::gemm_scratch<T> scratch;
    detail::syrk_lower_strips(C, ldc, A, lda, alpha, beta, rows, columns, gemm_config<T>::mc,
                              scratch.get());
}

/// @brief Contiguous SYRK, lower: \f$C \leftarrow \alpha A A^T + \beta C\f$.
template <std::floating_point T>
inline void syrk_lower(T *NUM_K_RESTRICT C, const T *NUM_K_RESTRICT A, T alpha, T beta, idx rows,
                       idx columns) noexcept {
    syrk_lower(C, rows, A, columns, alpha, beta, rows, columns);
}

/// @brief Transposed-left matrix product: \f$C \leftarrow \alpha A^T B + \beta C\f$.
template <std::floating_point T>
inline void gemm_transpose_left(T *NUM_K_RESTRICT C, idx ldc, const T *NUM_K_RESTRICT A, idx lda,
                                const T *NUM_K_RESTRICT B, idx ldb, T alpha, T beta, idx rows,
                                idx a_cols, idx b_cols) noexcept {
    const detail::gemm_scratch<T> scratch;
    detail::gemm_strided(C, ldc, A, idx{1}, lda, B, ldb, idx{1}, alpha, beta, a_cols, b_cols, rows,
                         scratch.get());
}

/// @brief Block projection: \f$h \leftarrow V^T w\f$.
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

/// @brief Block linear combination: \f$y \leftarrow \alpha V c + \beta y\f$.
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

/// @brief Matrix transpose, out of place: \f$B \leftarrow A^T\f$.
template <std::floating_point T>
NUM_K_AINLINE void transpose(T *NUM_K_RESTRICT B, const T *NUM_K_RESTRICT A, idx m,
                             idx n) noexcept {
    for (idx i = 0; i < m; ++i) {
        for (idx j = 0; j < n; ++j) {
            B[(j * m) + i] = A[(i * n) + j];
        }
    }
}

/// @brief Rank-1 update (GER): \f$A \leftarrow A + \alpha x y^T\f$.
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

/// @brief Rank-1 update (GER), contiguous: \f$A \leftarrow A + \alpha x y^T\f$.
template <std::floating_point T>
NUM_K_AINLINE void ger(T *NUM_K_RESTRICT A, const T *NUM_K_RESTRICT x, const T *NUM_K_RESTRICT y,
                       T alpha, idx m, idx n) noexcept {
    ger(A, n, x, y, alpha, m, n);
}

/// @brief Lower triangular solve (TRSV): \f$x \leftarrow L^{-1} b\f$.
/// `x` and `b` must not alias; see the `alias_safe` overload.
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

/// @brief Lower triangular solve, in place: \f$x \leftarrow L^{-1} x\f$.
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

/// @brief Lower triangular solve, alias-safe: \f$x \leftarrow L^{-1} b\f$.
/// `x` and `b` may alias.
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

/// @brief Diagonal block size of the blocked TRSM routines.
inline constexpr idx trsm_block = 64;

/// @brief Unblocked lower triangular solve: \f$X \leftarrow L^{-1} X\f$.
/// Element (i, j) of L is `L[i*row_stride + j*col_stride]`; `unit` takes the diagonal as 1.
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

/// @brief Unblocked transposed lower triangular solve: \f$X \leftarrow L^{-T} X\f$.
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

/// @brief Blocked lower triangular solve: \f$X \leftarrow L^{-1} X\f$.
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

/// @brief Blocked transposed lower triangular solve: \f$X \leftarrow L^{-T} X\f$.
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

/// @brief Rows per batch in `trsm_lower_transpose_right_block`.
inline constexpr idx trsm_row_batch = 16;

/// @brief Unblocked right-side transposed lower solve: \f$X \leftarrow X L^{-T}\f$.
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
            // Column j accumulates in registers across the k loop.
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

/// @brief Lower triangular solve, multiple right-hand sides (TRSM): \f$X \leftarrow L^{-1} X\f$.
template <std::floating_point T>
inline void trsm_lower_inplace(T *NUM_K_RESTRICT X, idx ldx, const T *NUM_K_RESTRICT L, idx n,
                               idx nrhs) noexcept {
    detail::trsm_lower_blocked(X, ldx, L, n, idx{1}, n, nrhs, false);
}

/// @brief Unit lower triangular solve (TRSM): \f$X \leftarrow L^{-1} X\f$.
/// The diagonal is taken as 1.
template <std::floating_point T>
inline void trsm_unit_lower_inplace(T *NUM_K_RESTRICT X, idx ldx, const T *NUM_K_RESTRICT L,
                                    idx ldl, idx n, idx nrhs) noexcept {
    detail::trsm_lower_blocked(X, ldx, L, ldl, idx{1}, n, nrhs, true);
}

/// @brief Transposed lower triangular solve (TRSM): \f$X \leftarrow L^{-T} X\f$.
template <std::floating_point T>
inline void trsm_lower_transpose_inplace(T *NUM_K_RESTRICT X, idx ldx, const T *NUM_K_RESTRICT L,
                                         idx n, idx nrhs) noexcept {
    detail::trsm_lower_transpose_blocked(X, ldx, L, n, idx{1}, n, nrhs, false);
}

/// @brief Transposed unit lower triangular solve (TRSM): \f$X \leftarrow L^{-T} X\f$.
/// The diagonal is taken as 1.
template <std::floating_point T>
inline void trsm_unit_lower_transpose_inplace(T *NUM_K_RESTRICT X, idx ldx,
                                              const T *NUM_K_RESTRICT L, idx ldl, idx n,
                                              idx nrhs) noexcept {
    detail::trsm_lower_transpose_blocked(X, ldx, L, ldl, idx{1}, n, nrhs, true);
}

/// @brief Upper triangular solve (TRSM): \f$X \leftarrow U^{-1} X\f$.
template <std::floating_point T>
inline void trsm_upper_inplace(T *NUM_K_RESTRICT X, idx ldx, const T *NUM_K_RESTRICT U, idx ldu,
                               idx n, idx nrhs, bool unit = false) noexcept {
    detail::trsm_lower_transpose_blocked(X, ldx, U, idx{1}, ldu, n, nrhs, unit);
}

/// @brief Transposed upper triangular solve (TRSM): \f$X \leftarrow U^{-T} X\f$.
template <std::floating_point T>
inline void trsm_upper_transpose_inplace(T *NUM_K_RESTRICT X, idx ldx, const T *NUM_K_RESTRICT U,
                                         idx ldu, idx n, idx nrhs, bool unit = false) noexcept {
    detail::trsm_lower_blocked(X, ldx, U, idx{1}, ldu, n, nrhs, unit);
}

/// @brief Right-side transposed lower triangular solve (TRSM): \f$X \leftarrow X L^{-T}\f$.
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

/// @brief Upper triangular solve (TRSV): \f$x \leftarrow U^{-1} b\f$.
/// `x` and `b` must not alias; see the `alias_safe` overload.
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

/// @brief Upper triangular solve, in place: \f$x \leftarrow U^{-1} x\f$.
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

/// @brief Upper triangular solve, alias-safe: \f$x \leftarrow U^{-1} b\f$.
/// `x` and `b` may alias.
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

/// @brief Row swap: \f$A_{r_1,:} \leftrightarrow A_{r_2,:}\f$.
template <typename T>
NUM_K_AINLINE void swap_rows(T *NUM_K_RESTRICT A, idx lda, idx r1, idx r2, idx n) noexcept {
    if (r1 != r2) {
        swap(A + (r1 * lda), A + (r2 * lda), n);
    }
}

/// @brief Transposed lower triangular solve, in place: \f$x \leftarrow L^{-T} x\f$.
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

/// @brief Banded matrix-vector product (GBMV): \f$y \leftarrow \alpha A x + \beta y\f$.
/// A is in LAPACK band storage with `kl` sub- and `ku` superdiagonals.
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
