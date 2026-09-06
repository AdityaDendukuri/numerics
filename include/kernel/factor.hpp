/// @file kernel/factor.hpp
/// @brief Raw-pointer factorizations: Cholesky and LU, over T* and a dimension.
///
/// SPDX-License-Identifier: MIT
/// Part of numerics, (c) 2026 Aditya Dendukuri.
/// https://github.com/AdityaDendukuri/numerics
///
/// This file has no dependencies outside the standard library beyond
/// kernel/vector.hpp and kernel/dense.hpp: copy the three into another project
/// as-is, or lift a single routine out of it. Please keep the two attribution
/// lines above with whatever you take.
///
/// These are the factorizations expressed the way a consuming project can
/// actually use them: over `T *` and a dimension, with no owning container, no
/// backend dispatch and no link dependency. A project with its own matrix type
/// passes `A.data()`; a project using `std::vector<T>` passes `v.data()`.
///
/// The library's own `num::cholesky` and friends are thin wrappers over these, so
/// there is one implementation of each algorithm rather than a container-coupled
/// copy and a raw copy that drift apart.
///
/// Storage is row-major throughout, matching `num::basic_mat`.
#pragma once

#include "kernel/dense.hpp"
#include "kernel/vector.hpp"
#include <algorithm>
#include <cmath>
#include <concepts>

namespace num::kernel {

/// @brief Cholesky factorization \f$A = L L^T\f$ for symmetric positive definite \f$A\f$.
///
/// Writes the lower triangular factor into `L` (row-major, \f$n \times n\f$) and
/// zeroes the strict upper triangle. `L` and `A` may alias.
///
/// Returns false as soon as a non-positive pivot appears, which is the exact point
/// at which \f$A\f$ is shown not to be positive definite — the factorization is the
/// definitive test, where sampling the quadratic form is only evidence.
///
/// @param L Output lower triangular factor, n*n.
/// @param A Input symmetric matrix, n*n; only the lower triangle is read.
/// @param n mat dimension.
template <std::floating_point T>
[[nodiscard]] inline bool cholesky(T *L, const T *A, idx n) noexcept {
    for (idx i = 0; i < n; ++i) {
        for (idx j = 0; j <= i; ++j) {
            T sum = A[(i * n) + j];
            for (idx k = 0; k < j; ++k) {
                sum -= L[(i * n) + k] * L[(j * n) + k];
            }
            if (i == j) {
                if (!(sum > T(0))) {
                    return false;
                }
                L[(i * n) + j] = std::sqrt(sum);
            } else {
                L[(i * n) + j] = sum / L[(j * n) + j];
            }
        }
        for (idx j = i + 1; j < n; ++j) {
            L[(i * n) + j] = T(0);
        }
    }
    return true;
}

/// @brief In-place blocked Cholesky factorization, `A <- L` with `A = L*L^T`.
///
/// The lower triangle is factored in panels.  The panel solve and trailing
/// update are delegated to the raw TRSM/SYRK kernels so larger problems expose
/// the same computational spine as the standalone dense primitives.
template <std::floating_point T>
[[nodiscard]] inline bool cholesky_blocked(T *A, idx n, idx block_size = 64) noexcept {
    if (n == 0) {
        return true;
    }
    if (block_size == 0) {
        block_size = 1;
    }
    for (idx kk = 0; kk < n; kk += block_size) {
        const idx kb = std::min(block_size, n - kk);

        // L11 <- chol(A11), retaining the parent row stride.
        for (idx i = 0; i < kb; ++i) {
            for (idx j = 0; j <= i; ++j) {
                T sum = A[((kk + i) * n) + (kk + j)];
                for (idx p = 0; p < j; ++p) {
                    sum -= A[((kk + i) * n) + (kk + p)] * A[((kk + j) * n) + (kk + p)];
                }
                if (i == j) {
                    if (!(sum > T(0))) {
                        return false;
                    }
                    A[((kk + i) * n) + (kk + j)] = std::sqrt(sum);
                } else {
                    A[((kk + i) * n) + (kk + j)] = sum / A[((kk + j) * n) + (kk + j)];
                }
            }
            for (idx j = i + 1; j < kb; ++j) {
                A[((kk + i) * n) + (kk + j)] = T(0);
            }
        }

        const idx trailing = n - (kk + kb);
        if (trailing == 0) {
            continue;
        }

        // L21 <- A21 * L11^{-T}.
        trsm_lower_transpose_right_inplace(A + ((kk + kb) * n) + kk, n,
                                           A + (kk * n) + kk, n, trailing, kb);

        // A22 <- A22 - L21*L21^T (lower triangle only).
        syrk_lower(A + (((kk + kb) * n) + (kk + kb)), n,
                   A + (((kk + kb) * n) + kk), n, T(-1), T(1), trailing, kb);
    }
    // The packed lower factor is the public result; make the unused triangle explicit.
    for (idx i = 0; i < n; ++i) {
        for (idx j = i + 1; j < n; ++j) {
            A[(i * n) + j] = T(0);
        }
    }
    return true;
}

/// @brief Solve \f$A x = b\f$ from a Cholesky factor, via \f$L y = b\f$ then \f$L^T x = y\f$.
///
/// `x` and `b` may alias.
template <std::floating_point T>
inline void cholesky_solve(T *x, const T *L, const T *b, idx n) noexcept {
    // The public contract permits x == b, so select the explicitly alias-safe
    // raw path rather than making a false restrict promise to the compiler.
    trsv_lower(contract::alias_safe, x, L, b, n);
    trsv_transpose_lower(x, L, n, n);
}

/// @brief Factors a batch of independent small SPD matrices.
///
/// Each matrix is row-major and begins `matrix_stride` elements after the
/// previous one.  `L` and `A` may be the same batch.  Per-system status avoids
/// throwing away successful factorizations when one member is not SPD.
template <std::floating_point T>
[[nodiscard]] inline bool cholesky_batched(T *L, const T *A, idx n, idx batch_count,
                                           idx matrix_stride, bool *success = nullptr) noexcept {
    bool all_ok = true;
    for (idx batch = 0; batch < batch_count; ++batch) {
        const bool ok = cholesky(L + (batch * matrix_stride), A + (batch * matrix_stride), n);
        if (success != nullptr) {
            success[batch] = ok;
        }
        all_ok = all_ok && ok;
    }
    return all_ok;
}

/// @brief Solves a batch of independent systems from batched Cholesky factors.
/// `x` and `b` may be the same batch.
template <std::floating_point T>
inline void cholesky_solve_batched(T *x, const T *L, const T *b, idx n, idx batch_count,
                                   idx matrix_stride, idx vector_stride) noexcept {
    for (idx batch = 0; batch < batch_count; ++batch) {
        cholesky_solve(x + (batch * vector_stride), L + (batch * matrix_stride),
                       b + (batch * vector_stride), n);
    }
}

/// @brief LU factorization \f$PA = LU\f$ with partial pivoting, packed in place.
///
/// On return `LU` holds the unit-lower factor below the diagonal and the upper
/// factor on and above it; `piv[i]` is the row swapped with row i. Returns false
/// when an exactly zero pivot column is found, i.e. when A is singular.
///
/// @param LU In/out packed factors, n*n row-major; pass a copy of A.
/// @param piv Output pivot sequence, length n.
/// @param n mat dimension.
template <std::floating_point T, class Index>
[[nodiscard]] inline bool lu_factor(T *NUM_K_RESTRICT LU, Index *NUM_K_RESTRICT piv,
                                    idx n) noexcept {
    bool nonsingular = true;
    for (idx k = 0; k < n; ++k) {
        idx pivot_row = k;
        T best = std::abs(LU[(k * n) + k]);
        for (idx i = k + 1; i < n; ++i) {
            const T candidate = std::abs(LU[(i * n) + k]);
            if (candidate > best) {
                best = candidate;
                pivot_row = i;
            }
        }
        piv[k] = static_cast<Index>(pivot_row);

        if (pivot_row != k) {
            swap_rows(LU, n, k, pivot_row, n);
        }
        const T pivot = LU[(k * n) + k];
        if (pivot == T(0)) {
            nonsingular = false;
            continue;
        }
        for (idx i = k + 1; i < n; ++i) {
            const T factor = LU[(i * n) + k] / pivot;
            LU[(i * n) + k] = factor;
            for (idx j = k + 1; j < n; ++j) {
                LU[(i * n) + j] -= factor * LU[(k * n) + j];
            }
        }
    }
    return nonsingular;
}

/// @brief Blocked variant of `lu_factor`: same packed \f$PA = LU\f$ contract, panel-at-a-time.
template <std::floating_point T, class Index>
[[nodiscard]] inline bool lu_factor_blocked(T *NUM_K_RESTRICT LU, Index *NUM_K_RESTRICT piv, idx n,
                                            idx block_size = 64) noexcept {
    if (block_size == 0) block_size = 1;
    bool nonsingular = true;
    for (idx kk = 0; kk < n; kk += block_size) {
        const idx kb = std::min(block_size, n - kk), panel_end = kk + kb;
        for (idx k = kk; k < panel_end; ++k) {
            idx pivot_row = k;
            T best = std::abs(LU[(k * n) + k]);
            for (idx i = k + 1; i < n; ++i) {
                const T candidate = std::abs(LU[(i * n) + k]);
                if (candidate > best) { best = candidate; pivot_row = i; }
            }
            piv[k] = static_cast<Index>(pivot_row);
            if (pivot_row != k) swap_rows(LU, n, k, pivot_row, n);
            const T pivot = LU[(k * n) + k];
            if (pivot == T(0)) { nonsingular = false; continue; }
            for (idx i = k + 1; i < n; ++i) {
                const T factor = LU[(i * n) + k] / pivot;
                LU[(i * n) + k] = factor;
                for (idx j = k + 1; j < panel_end; ++j)
                    LU[(i * n) + j] -= factor * LU[(k * n) + j];
            }
        }
        const idx trailing = n - panel_end;
        if (trailing == 0) continue;
        // U12 <- L11^{-1} A12.
        trsm_unit_lower_inplace(LU + (kk * n) + panel_end, n, LU + (kk * n) + kk, n, kb, trailing);
        // A22 <- A22 - L21*U12.
        gemm(LU + (panel_end * n) + panel_end, n, LU + (panel_end * n) + kk, n,
             LU + (kk * n) + panel_end, n, T(-1), T(1), trailing, trailing, kb);
    }
    return nonsingular;
}

/// @brief Solve \f$A x = b\f$ from a packed \f$PA = LU\f$ factorization.
template <std::floating_point T, class Index>
inline void lu_solve(T *NUM_K_RESTRICT x, const T *NUM_K_RESTRICT LU,
                     const Index *NUM_K_RESTRICT piv, const T *b, idx n) noexcept {
    for (idx i = 0; i < n; ++i) {
        x[i] = b[i];
    }
    for (idx k = 0; k < n; ++k) {
        const idx p = static_cast<idx>(piv[k]);
        if (p != k) {
            const T tmp = x[k];
            x[k] = x[p];
            x[p] = tmp;
        }
    }
    // Unit-lower forward substitution, then upper back substitution.
    for (idx i = 1; i < n; ++i) {
        T sum = x[i];
        for (idx j = 0; j < i; ++j) {
            sum -= LU[(i * n) + j] * x[j];
        }
        x[i] = sum;
    }
    for (idx i = n; i-- > 0;) {
        T sum = x[i];
        for (idx j = i + 1; j < n; ++j) {
            sum -= LU[(i * n) + j] * x[j];
        }
        x[i] = sum / LU[(i * n) + i];
    }
}

/// @brief Invert packed LU factors in place: A <- A^{-1}.
template <std::floating_point T, class Index>
inline void lu_invert(T *NUM_K_RESTRICT LU, const Index *NUM_K_RESTRICT piv, idx n,
                      T *NUM_K_RESTRICT work) noexcept {
    for (idx col = 0; col < n; ++col) {
        for (idx i = 0; i < n; ++i) work[i] = (i == col) ? T(1) : T(0);
        for (idx k = 0; k < n; ++k) {
            const idx p = static_cast<idx>(piv[k]);
            if (p != k) std::swap(work[k], work[p]);
        }
        for (idx i = 1; i < n; ++i) {
            T sum = work[i];
            for (idx j = 0; j < i; ++j) sum -= LU[(i * n) + j] * work[j];
            work[i] = sum;
        }
        for (idx i = n; i-- > 0;) {
            T sum = work[i];
            for (idx j = i + 1; j < n; ++j) sum -= LU[(i * n) + j] * work[j];
            work[i] = sum / LU[(i * n) + i];
        }
        for (idx i = 0; i < n; ++i) {
            LU[(i * n) + col] = work[i];
        }
    }
}

/// @brief Invert lower Cholesky factor in place to produce A^{-1} = L^{-T} L^{-1}.
template <std::floating_point T>
inline void cholesky_invert(T *NUM_K_RESTRICT L, idx n, T *NUM_K_RESTRICT work) noexcept {
    for (idx j = 0; j < n; ++j) {
        L[(j * n) + j] = T(1) / L[(j * n) + j];
        for (idx i = j + 1; i < n; ++i) {
            T sum = T(0);
            for (idx k = j; k < i; ++k) {
                sum -= L[(i * n) + k] * L[(k * n) + j];
            }
            L[(i * n) + j] = sum / L[(i * n) + i];
        }
    }
    for (idx i = 0; i < n; ++i) {
        for (idx j = 0; j <= i; ++j) {
            T sum = T(0);
            for (idx k = i; k < n; ++k) {
                sum += L[(k * n) + i] * L[(k * n) + j];
            }
            work[(i * n) + j] = sum;
            work[(j * n) + i] = sum;
        }
    }
    std::copy_n(work, n * n, L);
}

/// @brief banded LU factorization with partial pivoting over LAPACK-compatible band storage.
template <std::floating_point T, class Index>
[[nodiscard]] inline bool banded_factor(T *NUM_K_RESTRICT ab, idx ldab, idx n, idx kl, idx ku,
                                        Index *NUM_K_RESTRICT ipiv) noexcept {
    for (idx i = 0; i < n; ++i) ipiv[i] = static_cast<Index>(i);
    const idx kv = ku + kl;
    for (idx j = 0; j < n; ++j) {
        const idx last_row = std::min(j + kl, n - 1);
        T max_val = std::abs(ab[kv + (j * ldab)]);
        idx pivot = j;
        for (idx i = j + 1; i <= last_row; ++i) {
            T val = std::abs(ab[kv + i - j + (j * ldab)]);
            if (val > max_val) { max_val = val; pivot = i; }
        }
        if (max_val == T(0)) return false;
        ipiv[j] = static_cast<Index>(pivot);
        if (pivot != j) {
            const idx col_start = (j > ku) ? j - ku : 0;
            const idx col_end = std::min(j + kl, n - 1);
            for (idx col = col_start; col <= col_end; ++col) {
                idx band_j = kl + ku + j - col;
                idx band_pivot = kl + ku + pivot - col;
                if (band_j < ldab && band_pivot < ldab) {
                    std::swap(ab[band_j + (col * ldab)], ab[band_pivot + (col * ldab)]);
                }
            }
        }
        T pivot_val = ab[kv + (j * ldab)];
        const idx num_rows = std::min(kl, n - j - 1);
        if (num_rows > 0 && pivot_val != T(0)) {
            T inv_pivot = T(1) / pivot_val;
            for (idx i = 1; i <= num_rows; ++i) {
                ab[kv + i + (j * ldab)] *= inv_pivot;
            }
            const idx col_end = std::min(j + ku, n - 1);
            for (idx k = j + 1; k <= col_end; ++k) {
                T akj = ab[kv + j - k + (k * ldab)];
                if (akj != T(0)) {
                    for (idx i = 1; i <= num_rows; ++i) {
                        idx band_row = kv + j + i - k;
                        if (band_row < ldab) {
                            ab[band_row + (k * ldab)] -= ab[kv + i + (j * ldab)] * akj;
                        }
                    }
                }
            }
        }
    }
    return true;
}

/// @brief banded LU solve for x = b over LAPACK-compatible factored band storage.
template <std::floating_point T, class Index>
inline void banded_solve(T *x, const T *NUM_K_RESTRICT ab, idx ldab, idx n, idx kl, idx ku,
                         const Index *NUM_K_RESTRICT ipiv) noexcept {
    const idx kv = ku + kl;
    for (idx i = 0; i < n; ++i) {
        if (static_cast<idx>(ipiv[i]) != i) {
            std::swap(x[i], x[static_cast<idx>(ipiv[i])]);
        }
    }
    for (idx j = 0; j < n; ++j) {
        if (x[j] != T(0)) {
            const idx last = std::min(j + kl, n - 1);
            T xj = x[j];
            for (idx i = j + 1; i <= last; ++i) {
                x[i] -= ab[kv + i - j + (j * ldab)] * xj;
            }
        }
    }
    for (idx j = n; j > 0; --j) {
        const idx col = j - 1;
        x[col] /= ab[kv + (col * ldab)];
        if (x[col] != T(0)) {
            const idx first = (col > ku) ? col - ku : 0;
            T xc = x[col];
            for (idx i = first; i < col; ++i) {
                x[i] -= ab[kv + i - col + (col * ldab)] * xc;
            }
        }
    }
}

} // namespace num::kernel
