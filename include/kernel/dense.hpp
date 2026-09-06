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
#include <type_traits>

// Architectural vector registers on the target. Used only to size `gemm`'s
// register tile: the accumulators must live in registers for the whole inner
// loop, and a tile that overflows the file spills to the stack and loses far
// more than the blocking gained.
#if defined(__AVX512F__) || defined(__aarch64__) || defined(_M_ARM64)
#define NUM_K_VECTOR_REGISTERS 32
#else
#define NUM_K_VECTOR_REGISTERS 16
#endif

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

/// @brief Rows in `gemm`'s register tile.
inline constexpr idx gemm_tile_rows = 4;

/// @brief Columns in `gemm`'s register tile: whole vectors, filling half the
/// register file so the accumulators survive the inner loop without spilling.
template <std::floating_point T>
inline constexpr idx gemm_tile_cols =
    (NUM_K_VECTOR_REGISTERS / 2 / gemm_tile_rows) * (NUM_K_VECTOR_BYTES / sizeof(T));

/// @brief Bytes of the B panel to keep resident across one row sweep.
///
/// One mid-size L2 is the target. Measured on this tree at n = 512/1024/2048,
/// a 1 MiB budget gave 34.5/31.0/24.9 GFLOP/s against 30.8/27.7/20.1 at 256 KiB
/// and 35.1/28.4/23.4 at 4 MiB -- the smaller budget cuts reuse, the larger one
/// stops fitting.
inline constexpr idx gemm_panel_bytes = idx{1} << 20;

/// @brief Accumulate `C += alpha*A*B` over `p in [p0, p1)`, register-tiled.
template <std::floating_point T>
inline void gemm_panel(T *NUM_K_RESTRICT C, idx ldc, const T *NUM_K_RESTRICT A, idx lda,
                       const T *NUM_K_RESTRICT B, idx ldb, T alpha, idx m, idx n, idx p0,
                       idx p1) noexcept {
    constexpr idx mr = gemm_tile_rows;
    constexpr idx nr = gemm_tile_cols<T>;

    idx i = 0;
    for (; i + mr <= m; i += mr) {
        idx j = 0;
        for (; j + nr <= n; j += nr) {
            T acc[mr][nr];
            for (idx a = 0; a < mr; ++a) {
                NUM_K_IVDEP
                for (idx b = 0; b < nr; ++b) {
                    acc[a][b] = C[((i + a) * ldc) + j + b];
                }
            }
            for (idx p = p0; p < p1; ++p) {
                const T *NUM_K_RESTRICT b_row = B + (p * ldb) + j;
                for (idx a = 0; a < mr; ++a) {
                    const T a_val = alpha * A[((i + a) * lda) + p];
                    NUM_K_IVDEP
                    for (idx b = 0; b < nr; ++b) {
                        acc[a][b] += a_val * b_row[b];
                    }
                }
            }
            for (idx a = 0; a < mr; ++a) {
                NUM_K_IVDEP
                for (idx b = 0; b < nr; ++b) {
                    C[((i + a) * ldc) + j + b] = acc[a][b];
                }
            }
        }
        // Columns past the last whole tile: same row block, one column at a time.
        for (; j < n; ++j) {
            for (idx a = 0; a < mr; ++a) {
                T sum = C[((i + a) * ldc) + j];
                for (idx p = p0; p < p1; ++p) {
                    sum += alpha * A[((i + a) * lda) + p] * B[(p * ldb) + j];
                }
                C[((i + a) * ldc) + j] = sum;
            }
        }
    }
    // Rows past the last whole tile: the plain i-k-j loop, still vectorized
    // along j but without register reuse. At most `mr - 1` rows land here.
    for (; i < m; ++i) {
        T *NUM_K_RESTRICT c_row = C + (i * ldc);
        for (idx p = p0; p < p1; ++p) {
            const T a_val = alpha * A[(i * lda) + p];
            const T *NUM_K_RESTRICT b_row = B + (p * ldb);
            NUM_K_IVDEP
            for (idx j = 0; j < n; ++j) {
                c_row[j] += a_val * b_row[j];
            }
        }
    }
}

} // namespace detail

/// @brief Dense matrix product `C <- alpha*A*B + beta*C`, with row strides.
///
/// Register-tiled and cache-panelled (see above). A, B, and C must not overlap.
template <std::floating_point T>
inline void gemm(T *NUM_K_RESTRICT C, idx ldc, const T *NUM_K_RESTRICT A, idx lda,
                 const T *NUM_K_RESTRICT B, idx ldb, T alpha, T beta, idx m, idx n,
                 idx k) noexcept {
    if (beta == T(0)) {
        for (idx i = 0; i < m; ++i) {
            std::fill_n(C + (i * ldc), n, T(0));
        }
    } else if (beta != T(1)) {
        for (idx i = 0; i < m; ++i) {
            T *NUM_K_RESTRICT c_row = C + (i * ldc);
            NUM_K_IVDEP
            for (idx j = 0; j < n; ++j) {
                c_row[j] *= beta;
            }
        }
    }

    const idx kc = std::max<idx>(1, detail::gemm_panel_bytes / std::max<idx>(1, n * sizeof(T)));
    for (idx p0 = 0; p0 < k; p0 += kc) {
        detail::gemm_panel(C, ldc, A, lda, B, ldb, alpha, m, n, p0, std::min(p0 + kc, k));
    }
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
template <std::floating_point T>
inline void syrk_lower(T *NUM_K_RESTRICT C, idx ldc, const T *NUM_K_RESTRICT A, idx lda, T alpha,
                       T beta, idx rows, idx columns) noexcept {
    for (idx i = 0; i < rows; ++i) {
        T *NUM_K_RESTRICT c_row = C + (i * ldc);
        for (idx j = 0; j <= i; ++j) {
            const T *NUM_K_RESTRICT a_i = A + (i * lda);
            const T *NUM_K_RESTRICT a_j = A + (j * lda);
            const T sum = detail::reduce<T>(columns, [a_i, a_j](idx p) { return a_i[p] * a_j[p]; });
            c_row[j] = (alpha * sum) + (beta * c_row[j]);
        }
    }
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
    for (idx i = 0; i < a_cols; ++i) {
        T *NUM_K_RESTRICT c_row = C + (i * ldc);
        if (beta == T(0)) {
            fill(c_row, T(0), b_cols);
        } else if (beta != T(1)) {
            scale(c_row, beta, b_cols);
        }
        for (idx r = 0; r < rows; ++r) {
            const T a = alpha * A[(r * lda) + i];
            const T *NUM_K_RESTRICT b_row = B + (r * ldb);
            NUM_K_IVDEP
            for (idx j = 0; j < b_cols; ++j) {
                c_row[j] += a * b_row[j];
            }
        }
    }
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

/// @brief Solves `L*X=B` in-place for row-major `X` with `nrhs` columns.
///
/// Vectorization is across independent right-hand sides, making this the
/// natural primitive for block methods and batches sharing one factor.
template <std::floating_point T>
inline void trsm_lower_inplace(T *NUM_K_RESTRICT X, idx ldx, const T *NUM_K_RESTRICT L, idx n,
                               idx nrhs) noexcept {
    for (idx i = 0; i < n; ++i) {
        T *NUM_K_RESTRICT x_row = X + (i * ldx);
        for (idx j = 0; j < i; ++j) {
            const T lij = L[(i * n) + j];
            const T *NUM_K_RESTRICT solved_row = X + (j * ldx);
            NUM_K_IVDEP
            for (idx r = 0; r < nrhs; ++r) {
                x_row[r] -= lij * solved_row[r];
            }
        }
        const T inv_diag = T(1) / L[(i * n) + i];
        NUM_K_IVDEP
        for (idx r = 0; r < nrhs; ++r) {
            x_row[r] *= inv_diag;
        }
    }
}

/// @brief Solves `L^T*X=B` in-place for row-major multiple right-hand sides.
template <std::floating_point T>
inline void trsm_lower_transpose_inplace(T *NUM_K_RESTRICT X, idx ldx, const T *NUM_K_RESTRICT L,
                                         idx n, idx nrhs) noexcept {
    for (idx i = n; i-- > 0;) {
        T *row = X + (i * ldx);
        for (idx k = i + 1; k < n; ++k) {
            const T lik = L[(k * n) + i];
            const T *solved = X + (k * ldx);
            NUM_K_IVDEP
            for (idx r = 0; r < nrhs; ++r)
                row[r] -= lik * solved[r];
        }
        const T inv = T(1) / L[(i * n) + i];
        NUM_K_IVDEP
        for (idx r = 0; r < nrhs; ++r)
            row[r] *= inv;
    }
}

/// @brief Solves `X*L^T=B` in-place for row-major panel rows.
///
/// `X` is `rows x n`, `L` is lower triangular `n x n`; each row is an
/// independent right-side solve.  This is the panel solve in blocked
/// Cholesky: `L21 <- A21 * L11^{-T}`.
template <std::floating_point T>
inline void trsm_lower_transpose_right_inplace(T *NUM_K_RESTRICT X, idx ldx,
                                               const T *NUM_K_RESTRICT L, idx ldl, idx rows,
                                               idx n) noexcept {
    for (idx r = 0; r < rows; ++r) {
        T *NUM_K_RESTRICT row = X + (r * ldx);
        for (idx j = 0; j < n; ++j) {
            T s = row[j];
            for (idx k = 0; k < j; ++k) {
                s -= row[k] * L[(j * ldl) + k];
            }
            row[j] = s / L[(j * ldl) + j];
        }
    }
}

/// @brief Solves `L*X=B` in-place when `L` is unit lower triangular.
template <std::floating_point T>
inline void trsm_unit_lower_inplace(T *NUM_K_RESTRICT X, idx ldx, const T *NUM_K_RESTRICT L,
                                    idx ldl, idx n, idx nrhs) noexcept {
    for (idx i = 0; i < n; ++i) {
        T *NUM_K_RESTRICT row = X + (i * ldx);
        for (idx k = 0; k < i; ++k) {
            const T lik = L[(i * ldl) + k];
            const T *NUM_K_RESTRICT solved = X + (k * ldx);
            NUM_K_IVDEP
            for (idx j = 0; j < nrhs; ++j) {
                row[j] -= lik * solved[j];
            }
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
