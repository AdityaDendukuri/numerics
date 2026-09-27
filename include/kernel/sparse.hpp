/// @file kernel/sparse.hpp
/// @brief Raw-pointer kernels: CSR SpMV/SpMM and ILU(0) factorization.
///
/// SPDX-License-Identifier: MIT
/// Part of numerics, (c) 2026 Aditya Dendukuri.
/// https://github.com/AdityaDendukuri/numerics
///
/// Depends only on kernel/vector.hpp, and does not allocate. Keep the two attribution lines
/// above with whatever you copy.
#pragma once

#include "kernel/vector.hpp"
#include <cmath>
#include <concepts>
#include <type_traits>

namespace num::kernel {

namespace detail {

/// @brief Row length below which a plain running sum beats `reduce`'s blocked accumulation.
/// One accumulator block is the cutoff; a five-point stencil row measured about 15% slower
/// through `reduce`.
template <std::floating_point T>
inline constexpr idx short_row_cutoff = 4 * (NUM_K_VECTOR_BYTES / sizeof(T));

/// @brief One CSR row: plain running sum when short, blocked reduction when long.
template <std::floating_point T, std::integral Index>
[[nodiscard]] NUM_K_AINLINE T csr_row_dot(const T *NUM_K_RESTRICT val,
                                          const Index *NUM_K_RESTRICT col_idx,
                                          const T *NUM_K_RESTRICT x, Index start,
                                          idx length) noexcept {
    if (length < short_row_cutoff<T>) {
        T sum = T(0);
        for (idx k = 0; k < length; ++k) {
            const Index p = start + static_cast<Index>(k);
            sum += val[p] * x[col_idx[p]];
        }
        return sum;
    }
    return reduce<T>(length, [val, col_idx, x, start](idx k) {
        const Index p = start + static_cast<Index>(k);
        return val[p] * x[col_idx[p]];
    });
}

} // namespace detail

/// @brief Compressed Sparse Row (CSR) matrix-vector multiplication \f$\mathbf{y} \leftarrow A
/// \mathbf{x}\f$.
template <std::floating_point T, std::integral Index>
NUM_K_AINLINE void spmv(T *NUM_K_RESTRICT y, const T *NUM_K_RESTRICT val,
                        const Index *NUM_K_RESTRICT row_ptr, const Index *NUM_K_RESTRICT col_idx,
                        const T *NUM_K_RESTRICT x, std::type_identity_t<Index> m) noexcept {
    for (Index i = 0; i < m; ++i) {
        const Index start = row_ptr[i];
        const idx length = static_cast<idx>(row_ptr[i + 1] - start);
        y[i] = detail::csr_row_dot<T, Index>(val, col_idx, x, start, length);
    }
}

// Sparse incomplete factorization
//
// ILU(0) keeps A's pattern exactly, so it rewrites the value array in place. L's unit
// diagonal is implicit, and L's strict lower part shares the array with U.

/// @brief Locate each row's diagonal entry in a CSR pattern.
///
/// Requires column indices sorted within each row.
///
/// @param diagonal Output, size n: index into `col_idx` of entry (i,i).
/// @param row_ptr CSR row offsets, size n+1.
/// @param col_idx CSR column indices, sorted within each row.
/// @param n Number of rows.
/// @return False if some row has no diagonal entry, which ILU(0) cannot proceed without.
template <std::integral Index>
[[nodiscard]] NUM_K_AINLINE bool csr_diagonal_positions(Index *NUM_K_RESTRICT diagonal,
                                                        const Index *NUM_K_RESTRICT row_ptr,
                                                        const Index *NUM_K_RESTRICT col_idx,
                                                        std::type_identity_t<Index> n) noexcept {
    for (Index i = 0; i < n; ++i) {
        const Index end = row_ptr[i + 1];
        Index found = end;
        for (Index k = row_ptr[i]; k < end; ++k) {
            if (col_idx[k] == i) {
                found = k;
                break;
            }
        }
        if (found == end) {
            return false;
        }
        diagonal[i] = found;
    }
    return true;
}

/// @brief In-place ILU(0) factorization of a CSR value array.
///
/// `scratch` maps columns to positions in the row being eliminated, so matching a pivot row's
/// column is O(1). It is cleared on exit from each row.
///
/// @param val CSR values, overwritten with the combined factors.
/// @param row_ptr CSR row offsets, size n+1. Not modified.
/// @param col_idx CSR column indices, sorted within each row. Not modified.
/// @param diagonal Diagonal positions from `csr_diagonal_positions`.
/// @param scratch Workspace of size n; contents on entry and exit are irrelevant.
/// @param n Number of rows.
/// @return False if a pivot was zero or non-finite, leaving `val` partially overwritten.
template <std::floating_point T, std::integral Index>
[[nodiscard]] inline bool
ilu0_factor(T *NUM_K_RESTRICT val, const Index *NUM_K_RESTRICT row_ptr,
            const Index *NUM_K_RESTRICT col_idx, const Index *NUM_K_RESTRICT diagonal,
            Index *NUM_K_RESTRICT scratch, std::type_identity_t<Index> n) noexcept {
    constexpr Index unmarked = static_cast<Index>(-1);
    for (Index i = 0; i < n; ++i) {
        scratch[i] = unmarked;
    }

    for (Index i = 0; i < n; ++i) {
        const Index row_begin = row_ptr[i];
        const Index row_end = row_ptr[i + 1];
        for (Index k = row_begin; k < row_end; ++k) {
            scratch[col_idx[k]] = k;
        }

        // Columns strictly left of the diagonal are the L part of this row.
        for (Index k = row_begin; k < diagonal[i]; ++k) {
            const Index j = col_idx[k];
            const T pivot = val[diagonal[j]];
            if (pivot == T(0) || !std::isfinite(pivot)) {
                return false;
            }
            const T multiplier = val[k] / pivot;
            val[k] = multiplier;
            // only columns already in row i: this is what makes it ILU(0)
            for (Index p = diagonal[j] + 1; p < row_ptr[j + 1]; ++p) {
                const Index target = scratch[col_idx[p]];
                if (target != unmarked) {
                    val[target] -= multiplier * val[p];
                }
            }
        }

        const T pivot = val[diagonal[i]];
        if (pivot == T(0) || !std::isfinite(pivot)) {
            return false;
        }
        for (Index k = row_begin; k < row_end; ++k) {
            scratch[col_idx[k]] = unmarked;
        }
    }
    return true;
}

/// @brief Solve \f$LUx = b\f$ for factors packed by `ilu0_factor`.
///
/// Forward substitution against the implicit unit-diagonal L, then backward
/// substitution against U. `x` may alias `b`.
template <std::floating_point T, std::integral Index>
NUM_K_AINLINE void
csr_lu_solve(T *NUM_K_RESTRICT x, const T *NUM_K_RESTRICT val, const Index *NUM_K_RESTRICT row_ptr,
             const Index *NUM_K_RESTRICT col_idx, const Index *NUM_K_RESTRICT diagonal, const T *b,
             std::type_identity_t<Index> n) noexcept {
    for (Index i = 0; i < n; ++i) {
        T sum = b[i];
        for (Index k = row_ptr[i]; k < diagonal[i]; ++k) {
            sum -= val[k] * x[col_idx[k]];
        }
        x[i] = sum; // L has a unit diagonal, so no division here
    }
    for (Index i = n; i-- > 0;) {
        T sum = x[i];
        for (Index k = diagonal[i] + 1; k < row_ptr[i + 1]; ++k) {
            sum -= val[k] * x[col_idx[k]];
        }
        x[i] = sum / val[diagonal[i]];
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
        const Index start = row_ptr[i];
        const idx length = static_cast<idx>(row_ptr[i + 1] - start);
        const T s = detail::csr_row_dot<T, Index>(val, col_idx, x, start, length);
        y[i] = (alpha * s) + (beta * y[i]);
    }
}

/// @brief CSR sparse matrix times a row-major dense block, `Y <- A*X`, with `nrhs` values per
/// row. The right-hand-side loop is innermost, so it vectorizes.
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

} // namespace num::kernel
