/// @file kernel/sparse.hpp
/// @brief Raw-pointer kernels: CSR SpMV/SpMM and ILU(0) factorization.
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
/// Kernels assume non-owning, caller-sized buffers and do not allocate.
#pragma once

#include "kernel/vector.hpp"
#include <cmath>
#include <concepts>
#include <type_traits>

namespace num::kernel {

namespace detail {

/// @brief Row length below which `reduce`'s blocked accumulation costs more than
/// it saves, so a plain running sum wins.
///
/// `reduce` splits a range across several vector accumulators and combines them
/// pairwise at the end. That is a large win on a long run and a pure loss on a
/// row too short to fill even one accumulator block: the setup and the final
/// combine still happen, but no lane is ever reused. A five-point stencil — the
/// single most common sparse pattern there is — has five entries per row, and
/// measured ~15% slower through `reduce` than through a plain loop. One
/// accumulator block is the natural cutoff.
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
// ILU(0) computes L and U whose product approximates A while reusing A's own
// sparsity pattern exactly: no fill-in is admitted, so the factors cost the same
// storage as the matrix and the whole computation is a rewrite of the value
// array in place. The unit diagonal of L is implicit, so L's strict lower part
// and all of U share one array, distinguished by the position of each row's
// diagonal.
//
// These take a caller-supplied `diagonal` index and scratch buffer rather than
// allocating, in keeping with the rest of this file.

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
/// On success `val` holds the strict lower part of L and the whole of U, sharing
/// the original pattern; L's diagonal is unit and not stored. The pattern arrays
/// are untouched.
///
/// The inner update needs, for each entry of the pivot row, the matching column
/// in the current row. Searching for it would make the factorization quadratic
/// in the row length, so `scratch` holds a column-to-position map for the row
/// being eliminated: written on entry to the row, cleared on exit, so its state
/// never leaks between rows.
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
            // Subtract multiplier times the U part of row j, but only where the
            // column already exists in row i: that restriction is what makes
            // this ILU(0) rather than a complete factorization.
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

} // namespace num::kernel
