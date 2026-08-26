/// @file kernel/raw.hpp
/// @brief Raw-pointer inline kernels: BLAS-1/2, CSR, Householder, Givens, triangular solves.
///
/// SPDX-License-Identifier: MIT
/// Part of numerics, (c) 2026 Aditya Dendukuri.
/// https://github.com/AdityaDendukuri/numerics
///
/// This file has no dependencies outside the standard library: copy it into
/// another project as-is, or lift a single routine out of it together with the
/// NUM_K_* macro block below. Please keep the two attribution lines above with
/// whatever you take.
///
/// Kernels assume non-owning, caller-sized buffers and do not allocate.
#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <concepts>
#include <cstddef>
#include <type_traits>

namespace num {
/// Index type. Identical to the definition in container/types.hpp, and repeating
/// an identical typedef is well-formed, so this file stays self-contained whether
/// or not the rest of numerics is present.
using idx = std::size_t;
} // namespace num

#if defined(__GNUC__) || defined(__clang__)
#define NUM_K_AINLINE [[gnu::always_inline]] inline
#define NUM_K_RESTRICT __restrict__
#define NUM_K_IVDEP _Pragma("GCC ivdep")
#else
#define NUM_K_AINLINE inline
#define NUM_K_RESTRICT
#define NUM_K_IVDEP
#endif

namespace num::kernel::raw {

/// Computational contracts used at the unchecked kernel boundary.  The typed
/// layers are responsible for proving these preconditions before selecting a
/// raw overload; the tags make an intentional relaxation visible at the call
/// site without adding runtime state.
namespace contract {
struct alias_safe_t final {};
struct throughput_t final {};
struct ordered_t final {};

inline constexpr alias_safe_t alias_safe{};
inline constexpr throughput_t throughput{};
inline constexpr ordered_t ordered{};
} // namespace contract

// Vector Level-1 BLAS & Fused Kernels

/// @brief Copies a vector, \f$y_i \leftarrow x_i\f$.
template <typename T>
NUM_K_AINLINE void copy(T *NUM_K_RESTRICT y, const T *NUM_K_RESTRICT x, idx n) noexcept {
    NUM_K_IVDEP
    for (idx i = 0; i < n; ++i) {
        y[i] = x[i];
    }
}

/// @brief Fills a vector, \f$x_i \leftarrow value\f$.
template <typename T>
NUM_K_AINLINE void fill(T *NUM_K_RESTRICT x, T value, idx n) noexcept {
    NUM_K_IVDEP
    for (idx i = 0; i < n; ++i) {
        x[i] = value;
    }
}

/// @brief Copies strided vectors, \f$y_i \leftarrow x_i\f$.
template <typename T>
NUM_K_AINLINE void copy_strided(T *NUM_K_RESTRICT y, idx incy, const T *NUM_K_RESTRICT x, idx incx,
                                idx n) noexcept {
    NUM_K_IVDEP
    for (idx i = 0; i < n; ++i) {
        y[i * incy] = x[i * incx];
    }
}

/// @brief Copies and scales strided vectors, \f$y_i \leftarrow \alpha x_i\f$.
template <std::floating_point T>
NUM_K_AINLINE void scale_copy_strided(T *NUM_K_RESTRICT y, idx incy, const T *NUM_K_RESTRICT x,
                                      idx incx, T alpha, idx n) noexcept {
    NUM_K_IVDEP
    for (idx i = 0; i < n; ++i) {
        y[i * incy] = alpha * x[i * incx];
    }
}

/// @brief Strided vector update, \f$y_i \leftarrow y_i + \alpha x_i\f$.
template <std::floating_point T>
NUM_K_AINLINE void axpy_strided(T *NUM_K_RESTRICT y, idx incy, const T *NUM_K_RESTRICT x, idx incx,
                                T alpha, idx n) noexcept {
    NUM_K_IVDEP
    for (idx i = 0; i < n; ++i) {
        y[i * incy] += alpha * x[i * incx];
    }
}

template <std::floating_point T>
NUM_K_AINLINE T norm_sq_strided(const T *NUM_K_RESTRICT x, idx incx, idx n) noexcept {
    T s = T(0);
    for (idx i = 0; i < n; ++i) s += x[i * incx] * x[i * incx];
    return s;
}

template <typename T>
NUM_K_AINLINE void swap_strided(T *x, idx incx, T *y, idx incy, idx n) noexcept {
    for (idx i = 0; i < n; ++i) std::swap(x[i * incx], y[i * incy]);
}

/// @brief Computes in-place scalar scaling \f$x_i \leftarrow \alpha x_i\f$.
template <std::floating_point T>
NUM_K_AINLINE void scale(T *NUM_K_RESTRICT x, T alpha, idx n) noexcept {
    NUM_K_IVDEP
    for (idx i = 0; i < n; ++i) {
        x[i] *= alpha;
    }
}

/// @brief Computes vector update \f$y_i \leftarrow y_i + \alpha x_i\f$ (BLAS AXPY).
template <std::floating_point T>
NUM_K_AINLINE void axpy(T *NUM_K_RESTRICT y, const T *NUM_K_RESTRICT x, T alpha, idx n) noexcept {
    NUM_K_IVDEP
    for (idx i = 0; i < n; ++i) {
        y[i] += alpha * x[i];
    }
}

/// @brief Computes fused scaled vector update \f$y_i \leftarrow a x_i + b y_i\f$ in a single memory
/// pass.
template <std::floating_point T>
NUM_K_AINLINE void axpby(T *NUM_K_RESTRICT y, const T *NUM_K_RESTRICT x, T a, T b, idx n) noexcept {
    NUM_K_IVDEP
    for (idx i = 0; i < n; ++i) {
        y[i] = (a * x[i]) + (b * y[i]);
    }
}

/// @brief Computes vector linear combination \f$z_i \leftarrow a x_i + b y_i\f$.
template <std::floating_point T>
NUM_K_AINLINE void axpbyz(T *NUM_K_RESTRICT z, const T *NUM_K_RESTRICT x, const T *NUM_K_RESTRICT y,
                          T a, T b, idx n) noexcept {
    NUM_K_IVDEP
    for (idx i = 0; i < n; ++i) {
        z[i] = (a * x[i]) + (b * y[i]);
    }
}

/// @brief Computes vector dot product \f$\mathbf{x} \cdot \mathbf{y} = \sum_{i=0}^{n-1} x_i y_i\f$.
template <std::floating_point T>
[[nodiscard]] NUM_K_AINLINE T dot(const T *NUM_K_RESTRICT x, const T *NUM_K_RESTRICT y,
                                  idx n) noexcept {
    T s = T(0);
    NUM_K_IVDEP
    for (idx i = 0; i < n; ++i) {
        s += x[i] * y[i];
    }
    return s;
}

/// @brief Throughput-oriented dot product with independent accumulation chains.
///
/// The grouping differs from the source-ordered overload and can therefore
/// differ by roundoff.  The explicit tag prevents a solver from silently
/// trading numerical semantics for instruction-level parallelism.
template <std::floating_point T>
[[nodiscard]] NUM_K_AINLINE T dot(contract::throughput_t, const T *NUM_K_RESTRICT x,
                                  const T *NUM_K_RESTRICT y, idx n) noexcept {
    T s0 = T(0), s1 = T(0), s2 = T(0), s3 = T(0);
    idx i = 0;
    for (; i + 4 <= n; i += 4) {
        s0 += x[i] * y[i];
        s1 += x[i + 1] * y[i + 1];
        s2 += x[i + 2] * y[i + 2];
        s3 += x[i + 3] * y[i + 3];
    }
    T s = (s0 + s1) + (s2 + s3);
    for (; i < n; ++i) {
        s += x[i] * y[i];
    }
    return s;
}

/// @brief Explicitly selects the source-ordered dot-product contract.
template <std::floating_point T>
[[nodiscard]] NUM_K_AINLINE T dot(contract::ordered_t, const T *NUM_K_RESTRICT x,
                                  const T *NUM_K_RESTRICT y, idx n) noexcept {
    return dot(x, y, n);
}

/// @brief Computes two inner products sharing the same left operand in one pass.
///
/// This is the primitive needed by fused Krylov recurrences: `x` is loaded once
/// instead of once per reduction.  The result members are deliberately named so
/// their mathematical meaning remains clear at call sites.
template <std::floating_point T>
struct dot2_result {
    T xy;
    T xz;
};

template <std::floating_point T>
[[nodiscard]] NUM_K_AINLINE dot2_result<T> dot2(const T *NUM_K_RESTRICT x,
                                                const T *NUM_K_RESTRICT y,
                                                const T *NUM_K_RESTRICT z, idx n) noexcept {
    T xy = T(0);
    T xz = T(0);
    NUM_K_IVDEP
    for (idx i = 0; i < n; ++i) {
        const T xi = x[i];
        xy += xi * y[i];
        xz += xi * z[i];
    }
    return {xy, xz};
}

/// @brief Computes an inner product and squared norm in one memory pass.
template <std::floating_point T>
struct dot_norm_result {
    T dot;
    T norm_sq;
};

template <std::floating_point T>
[[nodiscard]] NUM_K_AINLINE dot_norm_result<T>
dot_norm_sq(const T *NUM_K_RESTRICT x, const T *NUM_K_RESTRICT y, idx n) noexcept {
    T xy = T(0);
    T yy = T(0);
    NUM_K_IVDEP
    for (idx i = 0; i < n; ++i) {
        const T yi = y[i];
        xy += x[i] * yi;
        yy += yi * yi;
    }
    return {xy, yy};
}

/// @brief Updates `y <- y + alpha*x` and returns the new squared norm of `y`.
///
/// The update and convergence statistic share one traversal.  This is generally
/// bandwidth-optimal for iterative methods because the updated values remain in
/// registers for the reduction.
template <std::floating_point T>
[[nodiscard]] NUM_K_AINLINE T axpy_norm_sq(T *NUM_K_RESTRICT y, const T *NUM_K_RESTRICT x, T alpha,
                                           idx n) noexcept {
    T yy = T(0);
    NUM_K_IVDEP
    for (idx i = 0; i < n; ++i) {
        const T yi = y[i] + (alpha * x[i]);
        y[i] = yi;
        yy += yi * yi;
    }
    return yy;
}

/// @brief Returns \f$\|a x + b y\|_2^2\f$ without materializing the combination.
template <std::floating_point T>
[[nodiscard]] NUM_K_AINLINE T linear_combination_norm_sq(const T *NUM_K_RESTRICT x, T a,
                                                         const T *NUM_K_RESTRICT y, T b,
                                                         idx n) noexcept {
    T result = T(0);
    NUM_K_IVDEP
    for (idx i = 0; i < n; ++i) {
        const T value = (a * x[i]) + (b * y[i]);
        result += value * value;
    }
    return result;
}

/// @brief Computes squared Euclidean norm \f$\|\mathbf{x}\|_2^2 = \sum_{i=0}^{n-1} x_i^2\f$.
template <std::floating_point T>
[[nodiscard]] NUM_K_AINLINE T norm_sq(const T *NUM_K_RESTRICT x, idx n) noexcept {
    T s = T(0);
    NUM_K_IVDEP
    for (idx i = 0; i < n; ++i) {
        s += x[i] * x[i];
    }
    return s;
}

/// @brief Computes Euclidean \f$L_2\f$ norm \f$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=0}^{n-1} x_i^2}\f$.
template <std::floating_point T>
[[nodiscard]] NUM_K_AINLINE T norm(const T *NUM_K_RESTRICT x, idx n) noexcept {
    return std::sqrt(norm_sq(x, n));
}

/// @brief Computes \f$L_1\f$ norm \f$\|\mathbf{x}\|_1 = \sum_{i=0}^{n-1} |x_i|\f$.
template <std::floating_point T>
[[nodiscard]] NUM_K_AINLINE T l1_norm(const T *NUM_K_RESTRICT x, idx n) noexcept {
    T s = T(0);
    NUM_K_IVDEP
    for (idx i = 0; i < n; ++i) {
        s += std::abs(x[i]);
    }
    return s;
}

/// @brief Computes \f$L_\infty\f$ norm \f$\|\mathbf{x}\|_\infty = \max_{0 \le i < n} |x_i|\f$.
template <std::floating_point T>
[[nodiscard]] NUM_K_AINLINE T linf_norm(const T *NUM_K_RESTRICT x, idx n) noexcept {
    T mx = T(0);
    for (idx i = 0; i < n; ++i) {
        const T v = std::abs(x[i]);
        if (v > mx) {
            mx = v;
        }
    }
    return mx;
}

/// @brief Computes scalar vector sum \f$\sum_{i=0}^{n-1} x_i\f$.
template <std::floating_point T>
[[nodiscard]] NUM_K_AINLINE T sum(const T *NUM_K_RESTRICT x, idx n) noexcept {
    T s = T(0);
    NUM_K_IVDEP
    for (idx i = 0; i < n; ++i) {
        s += x[i];
    }
    return s;
}

/// @brief In-place vector element clamp \f$x_i \leftarrow \min(\max(x_i, x_{\min}), x_{\max})\f$.
template <std::floating_point T>
NUM_K_AINLINE void clamp(T *NUM_K_RESTRICT x, T lo, T hi, idx n) noexcept {
    NUM_K_IVDEP
    for (idx i = 0; i < n; ++i) {
        x[i] = std::clamp(x[i], lo, hi);
    }
}

/// @brief Computes elementwise Hadamard product \f$z_i = x_i \cdot y_i\f$.
template <std::floating_point T>
NUM_K_AINLINE void hadamard_mul(T *NUM_K_RESTRICT z, const T *NUM_K_RESTRICT x,
                                const T *NUM_K_RESTRICT y, idx n) noexcept {
    NUM_K_IVDEP
    for (idx i = 0; i < n; ++i) {
        z[i] = x[i] * y[i];
    }
}

/// @brief Computes elementwise division \f$z_i = x_i / y_i\f$.
template <std::floating_point T>
NUM_K_AINLINE void hadamard_div(T *NUM_K_RESTRICT z, const T *NUM_K_RESTRICT x,
                                const T *NUM_K_RESTRICT y, idx n) noexcept {
    NUM_K_IVDEP
    for (idx i = 0; i < n; ++i) {
        z[i] = x[i] / y[i];
    }
}

/// @brief Computes elementwise reciprocal \f$y_i = 1 / x_i\f$ (e.g. for Jacobi preconditioners).
template <std::floating_point T>
NUM_K_AINLINE void inv(T *NUM_K_RESTRICT y, const T *NUM_K_RESTRICT x, idx n) noexcept {
    NUM_K_IVDEP
    for (idx i = 0; i < n; ++i) {
        y[i] = T(1) / x[i];
    }
}

// Orthogonal Plane Transformations (Givens / Arnoldi / QR)

/// @brief Constructs Givens rotation parameters \f$(c, s)\f$ such that:
/// \f[
/// \begin{bmatrix} c & s \\ -s & c \end{bmatrix} \begin{bmatrix} a \\ b \end{bmatrix} =
/// \begin{bmatrix} r \\ 0 \end{bmatrix}, \qquad c^2 + s^2 = 1
/// \f]
template <std::floating_point T>
NUM_K_AINLINE void rotg(T a, T b, T &c, T &s) noexcept {
    if (b == T(0)) {
        c = T(1);
        s = T(0);
    } else if (a == T(0)) {
        c = T(0);
        s = T(1);
    } else {
        T r = std::hypot(a, b);
        c = a / r;
        s = b / r;
    }
}

/// @brief Applies Givens plane rotation in-place:
/// \f[
/// \begin{bmatrix} x_i \\ y_i \end{bmatrix} \leftarrow \begin{bmatrix} c & s \\ -s & c
/// \end{bmatrix} \begin{bmatrix} x_i \\ y_i \end{bmatrix}
/// \f]
template <std::floating_point T>
NUM_K_AINLINE void rot(T *NUM_K_RESTRICT x, T *NUM_K_RESTRICT y, T c, T s, idx n) noexcept {
    NUM_K_IVDEP
    for (idx i = 0; i < n; ++i) {
        T xi = x[i];
        T yi = y[i];
        x[i] = (c * xi) + (s * yi);
        y[i] = (-s * xi) + (c * yi);
    }
}

// Dense Matrix Level-2 BLAS & Triangular Solvers

/// @brief Dense matrix-vector multiplication \f$\mathbf{y} \leftarrow A \mathbf{x}\f$ for \f$A \in
/// \mathbb{R}^{m \times n}\f$.
template <std::floating_point T>
NUM_K_AINLINE void matvec(T *NUM_K_RESTRICT y, const T *NUM_K_RESTRICT A, const T *NUM_K_RESTRICT x,
                          idx m, idx n) noexcept {
    for (idx i = 0; i < m; ++i) {
        T s = T(0);
        const T *row = A + (i * n);
        NUM_K_IVDEP
        for (idx j = 0; j < n; ++j) {
            s += row[j] * x[j];
        }
        y[i] = s;
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

template <std::floating_point T>
NUM_K_AINLINE T column_dot(const T *NUM_K_RESTRICT A, idx lda, idx rows, idx p, idx q) noexcept {
    T s = T(0);
    for (idx i = 0; i < rows; ++i) s += A[(i * lda) + p] * A[(i * lda) + q];
    return s;
}

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

/// @brief Dense matrix product `C <- alpha*A*B + beta*C`, with row strides.
///
/// The i-k-j order reuses each element of A while streaming contiguous rows of
/// B and C.  It is the portable reference/mid-size implementation and the
/// computational building block for blocked factorizations.  A, B, and C must
/// not overlap.
template <std::floating_point T>
inline void gemm(T *NUM_K_RESTRICT C, idx ldc, const T *NUM_K_RESTRICT A, idx lda,
                 const T *NUM_K_RESTRICT B, idx ldb, T alpha, T beta, idx m, idx n,
                 idx k) noexcept {
    for (idx i = 0; i < m; ++i) {
        T *NUM_K_RESTRICT c_row = C + (i * ldc);
        if (beta == T(0)) {
            NUM_K_IVDEP
            for (idx j = 0; j < n; ++j) {
                c_row[j] = T(0);
            }
        } else if (beta != T(1)) {
            NUM_K_IVDEP
            for (idx j = 0; j < n; ++j) {
                c_row[j] *= beta;
            }
        }
        for (idx p = 0; p < k; ++p) {
            const T a = alpha * A[(i * lda) + p];
            const T *NUM_K_RESTRICT b_row = B + (p * ldb);
            NUM_K_IVDEP
            for (idx j = 0; j < n; ++j) {
                c_row[j] += a * b_row[j];
            }
        }
    }
}

template <std::floating_point T>
inline void gemm(T *NUM_K_RESTRICT C, const T *NUM_K_RESTRICT A, const T *NUM_K_RESTRICT B, T alpha,
                 T beta, idx m, idx n, idx k) noexcept {
    gemm(C, n, A, k, B, n, alpha, beta, m, n, k);
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
            T sum = T(0);
            const T *NUM_K_RESTRICT a_i = A + (i * lda);
            const T *NUM_K_RESTRICT a_j = A + (j * lda);
            for (idx p = 0; p < columns; ++p) {
                sum += a_i[p] * a_j[p];
            }
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
        T sum = T(0);
        for (idx j = 0; j < columns; ++j) {
            sum += v_row[j] * coefficients[j];
        }
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
inline void trsm_lower_transpose_inplace(T *NUM_K_RESTRICT X, idx ldx,
                                          const T *NUM_K_RESTRICT L, idx n, idx nrhs) noexcept {
    for (idx i = n; i-- > 0;) {
        T *row = X + (i * ldx);
        for (idx k = i + 1; k < n; ++k) {
            const T lik = L[(k * n) + i];
            const T *solved = X + (k * ldx);
            NUM_K_IVDEP
            for (idx r = 0; r < nrhs; ++r) row[r] -= lik * solved[r];
        }
        const T inv = T(1) / L[(i * n) + i];
        NUM_K_IVDEP
        for (idx r = 0; r < nrhs; ++r) row[r] *= inv;
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

// Sparse CSR Matrix Operations (Krylov & PDE Stencils)

/// @brief Compressed Sparse Row (CSR) matrix-vector multiplication \f$\mathbf{y} \leftarrow A
/// \mathbf{x}\f$.
template <std::floating_point T, std::integral Index>
NUM_K_AINLINE void spmv(T *NUM_K_RESTRICT y, const T *NUM_K_RESTRICT val,
                        const Index *NUM_K_RESTRICT row_ptr, const Index *NUM_K_RESTRICT col_idx,
                        const T *NUM_K_RESTRICT x, std::type_identity_t<Index> m) noexcept {
    for (Index i = 0; i < m; ++i) {
        T s = T(0);
        const Index start = row_ptr[i];
        const Index end = row_ptr[i + 1];
        NUM_K_IVDEP
        for (Index p = start; p < end; ++p) {
            s += val[p] * x[col_idx[p]];
        }
        y[i] = s;
    }
}

/// @brief Fused CSR SpMV and vector accumulation \f$\mathbf{y} \leftarrow \alpha A \mathbf{x} +
/// \beta \mathbf{y}\f$.
template <std::floating_point T, std::integral Index>
NUM_K_AINLINE void spmv_axpy(T *NUM_K_RESTRICT y, T alpha, const T *NUM_K_RESTRICT val,
                             const Index *NUM_K_RESTRICT row_ptr,
                             const Index *NUM_K_RESTRICT col_idx, const T *NUM_K_RESTRICT x, T beta,
                             std::type_identity_t<Index> m) noexcept {
    for (Index i = 0; i < m; ++i) {
        T s = T(0);
        const Index start = row_ptr[i];
        const Index end = row_ptr[i + 1];
        NUM_K_IVDEP
        for (Index p = start; p < end; ++p) {
            s += val[p] * x[col_idx[p]];
        }
        y[i] = (alpha * s) + (beta * y[i]);
    }
}

/// @brief CSR sparse matrix times a row-major dense block, `Y <- A*X`.
///
/// `X` and `Y` contain `nrhs` contiguous values per matrix row.  Traversing the
/// right-hand-side dimension innermost amortizes CSR index/value loads and gives
/// the compiler a regular SIMD loop even though the sparse row itself is irregular.
template <std::floating_point T, std::integral Index>
inline void spmm(T *NUM_K_RESTRICT Y, idx ldy, const T *NUM_K_RESTRICT val,
                 const Index *NUM_K_RESTRICT row_ptr, const Index *NUM_K_RESTRICT col_idx,
                 const T *NUM_K_RESTRICT X, idx ldx, std::type_identity_t<Index> m,
                 idx nrhs) noexcept {
    for (Index i = 0; i < m; ++i) {
        T *NUM_K_RESTRICT y_row = Y + (static_cast<idx>(i) * ldy);
        NUM_K_IVDEP
        for (idx r = 0; r < nrhs; ++r) {
            y_row[r] = T(0);
        }
        for (Index p = row_ptr[i]; p < row_ptr[i + 1]; ++p) {
            const T a = val[p];
            const T *NUM_K_RESTRICT x_row = X + (static_cast<idx>(col_idx[p]) * ldx);
            NUM_K_IVDEP
            for (idx r = 0; r < nrhs; ++r) {
                y_row[r] += a * x_row[r];
            }
        }
    }
}

// Orthogonal, Householder & Jacobi Rotations

/// @brief Computes elementary Householder reflector vector \f$\mathbf{v}\f$ and scalar \f$\beta\f$
/// such that:
/// \f[
/// (I - \beta \mathbf{v} \mathbf{v}^T) \mathbf{x} = \mp \|\mathbf{x}\|_2 \mathbf{e}_1
/// \f]
/// \f$\mathbf{v}\f$ is sized \f$m\f$, with \f$v_0 = 1\f$ implicitly assigned.
template <std::floating_point T>
NUM_K_AINLINE void householder_vector(T *NUM_K_RESTRICT v, T &beta, const T *NUM_K_RESTRICT x,
                                      idx m) noexcept {
    T sq = T(0);
    NUM_K_IVDEP
    for (idx i = 0; i < m; ++i) {
        sq += x[i] * x[i];
    }
    const T norm_x = std::sqrt(sq);
    if (norm_x < T(1e-15)) {
        beta = T(0);
        v[0] = T(1);
        return;
    }
    const T sign = (x[0] >= T(0)) ? T(1) : T(-1);
    const T mu = x[0] + (sign * norm_x);
    v[0] = T(1);
    T v_sq = T(1);
    NUM_K_IVDEP
    for (idx i = 1; i < m; ++i) {
        v[i] = x[i] / mu;
        v_sq += v[i] * v[i];
    }
    beta = T(2) / v_sq;
}

template <std::floating_point T>
NUM_K_AINLINE void householder_vector_strided(T *NUM_K_RESTRICT v, T &beta,
                                              const T *NUM_K_RESTRICT A, idx lda, idx offset,
                                              idx m) noexcept {
    for (idx i = 0; i < m; ++i) v[i] = A[((offset + i) * lda) + offset];
    householder_vector(v, beta, v, m);
}

template <std::floating_point T>
NUM_K_AINLINE void householder_left(T *NUM_K_RESTRICT A, idx lda, const T *NUM_K_RESTRICT v, T beta,
                                    idx m, idx n, T *NUM_K_RESTRICT work) noexcept;

/// @brief Compact Householder QR factorization; reflector tails remain below R's diagonal.
template <std::floating_point T>
inline void qr_factor_blocked(T *NUM_K_RESTRICT A, idx lda, idx m, idx n, T *NUM_K_RESTRICT tau,
                              T *NUM_K_RESTRICT v, T *NUM_K_RESTRICT work,
                              idx block_size = 32) noexcept {
    (void)block_size;
    const idx r = std::min(m, n);
    for (idx k = 0; k < r; ++k) {
        const idx len = m - k;
        T beta = T(0);
        householder_vector_strided(v, beta, A, lda, k, len);
        tau[k] = beta;
        if (beta == T(0)) continue;
        // R(k:m,k:n) <- H_k R(k:m,k:n).
        householder_left(A + (k * lda) + k, lda, v, beta, len, n - k, work);
        for (idx i = 1; i < len; ++i) A[((k + i) * lda) + k] = v[i];
    }
}

/// @brief Applies left Householder transformation \f$A \leftarrow (I - \beta \mathbf{v}
/// \mathbf{v}^T) A\f$ on an \f$m \times n\f$ block with stride `lda`. `work` is a caller-provided
/// scratch buffer of length at least \f$n\f$.
template <std::floating_point T>
NUM_K_AINLINE void householder_left(T *NUM_K_RESTRICT A, idx lda, const T *NUM_K_RESTRICT v, T beta,
                                    idx m, idx n, T *NUM_K_RESTRICT work) noexcept {
    NUM_K_IVDEP
    for (idx j = 0; j < n; ++j) {
        work[j] = T(0);
    }
    for (idx i = 0; i < m; ++i) {
        const T vi = v[i];
        const T *row = A + (i * lda);
        NUM_K_IVDEP
        for (idx j = 0; j < n; ++j) {
            work[j] += vi * row[j];
        }
    }
    NUM_K_IVDEP
    for (idx j = 0; j < n; ++j) {
        work[j] *= beta;
    }
    for (idx i = 0; i < m; ++i) {
        const T vi = v[i];
        T *row = A + (i * lda);
        NUM_K_IVDEP
        for (idx j = 0; j < n; ++j) {
            row[j] -= vi * work[j];
        }
    }
}

/// @brief Applies right Householder transformation \f$A \leftarrow A (I - \beta \mathbf{v}
/// \mathbf{v}^T)\f$ on an \f$m \times n\f$ block with stride `lda`.
template <std::floating_point T>
NUM_K_AINLINE void householder_right(T *NUM_K_RESTRICT A, idx lda, const T *NUM_K_RESTRICT v,
                                     T beta, idx m, idx n) noexcept {
    for (idx i = 0; i < m; ++i) {
        T *row = A + (i * lda);
        T dot_val = T(0);
        NUM_K_IVDEP
        for (idx j = 0; j < n; ++j) {
            dot_val += row[j] * v[j];
        }
        const T factor = beta * dot_val;
        NUM_K_IVDEP
        for (idx j = 0; j < n; ++j) {
            row[j] -= factor * v[j];
        }
    }
}

/// @brief Computes Jacobi rotation parameters \f$(c, s)\f$ annihilating off-diagonal entry
/// \f$A_{pq}\f$ in a symmetric \f$2 \times 2\f$ block.
template <std::floating_point T>
NUM_K_AINLINE void jacobi_rotation(T app, T aqq, T apq, T &c, T &s) noexcept {
    if (std::abs(apq) < T(1e-15)) {
        c = T(1);
        s = T(0);
        return;
    }
    const T tau = (aqq - app) / (T(2) * apq);
    const T t = std::copysign(T(1), tau) / (std::abs(tau) + std::sqrt(T(1) + (tau * tau)));
    c = T(1) / std::sqrt(T(1) + (t * t));
    s = c * t;
}

// Row / Column Operations & Swaps

/// @brief Swaps vector buffers \f$\mathbf{x} \leftrightarrow \mathbf{y}\f$ of length \f$n\f$
/// in-place.
template <typename T>
NUM_K_AINLINE void swap(T *NUM_K_RESTRICT x, T *NUM_K_RESTRICT y, idx n) noexcept {
    NUM_K_IVDEP
    for (idx i = 0; i < n; ++i) {
        std::swap(x[i], y[i]);
    }
}

/// @brief Swaps rows \f$r_1 \leftrightarrow r_2\f$ of length \f$n\f$ in row-major matrix \f$A\f$
/// with stride `lda`.
template <typename T>
NUM_K_AINLINE void swap_rows(T *NUM_K_RESTRICT A, idx lda, idx r1, idx r2, idx n) noexcept {
    if (r1 != r2) {
        swap(A + (r1 * lda), A + (r2 * lda), n);
    }
}

/// @brief Returns index of element with maximum absolute value \f$\arg\max_i |x_i|\f$ in vector
/// \f$\mathbf{x}\f$.
template <std::floating_point T>
[[nodiscard]] NUM_K_AINLINE idx argmax_abs(const T *NUM_K_RESTRICT x, idx n) noexcept {
    if (n == 0) {
        return 0;
    }
    idx best_idx = 0;
    T best_val = std::abs(x[0]);
    for (idx i = 1; i < n; ++i) {
        const T val = std::abs(x[i]);
        if (val > best_val) {
            best_val = val;
            best_idx = i;
        }
    }
    return best_idx;
}

// Triangular Solvers (Transposed & Advanced)

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

// Banded Matrix Kernels (LAPACK Column-Packed Storage)

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

/// @brief Mixed real-matrix, complex-vector product \f$x = Q y\f$.
///
/// Arises when a real orthogonal basis is applied to a complex Krylov coordinate
/// vector, as in projecting a resolvent solution back from the Krylov subspace.
/// Kept separate from `matvec` because the scalar types differ on the two sides.
template <std::floating_point T>
NUM_K_AINLINE void matvec_real_complex(std::complex<T> *NUM_K_RESTRICT x, const T *Q,
                                       const std::complex<T> *y, idx m, idx n) noexcept {
    for (idx i = 0; i < m; ++i) {
        const T *row = Q + (i * n);
        std::complex<T> sum{};
        for (idx j = 0; j < n; ++j) {
            sum += row[j] * y[j];
        }
        x[i] = sum;
    }
}

/// @brief Mixed transpose product \f$x = Q^T y\f$ with a real matrix and complex result.
///
/// The transpose companion to `matvec_real_complex`. Projecting a right-hand side
/// onto a real orthonormal basis produces complex coordinates when the right-hand
/// side is complex, and real ones widened to complex when it is not, so the input
/// scalar is a separate parameter.
///
/// @param x Output, length n.
/// @param Q Real matrix, m*n row-major.
/// @param y Input, length m, real or complex.
/// @param m Number of rows in Q.
/// @param n Number of columns in Q.
template <std::floating_point T, class In>
NUM_K_AINLINE void matvec_transpose_into_complex(std::complex<T> *NUM_K_RESTRICT x, const T *Q,
                                                 const In *y, idx m, idx n) noexcept {
    for (idx i = 0; i < n; ++i) {
        x[i] = std::complex<T>{};
    }
    for (idx j = 0; j < m; ++j) {
        const T *row = Q + (j * n);
        const std::complex<T> yj = y[j];
        for (idx i = 0; i < n; ++i) {
            x[i] += row[i] * yj;
        }
    }
}

} // namespace num::kernel::raw
