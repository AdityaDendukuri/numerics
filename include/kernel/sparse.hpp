/// @file kernel/sparse.hpp
/// @brief Raw-pointer kernels: CSR SpMV and SpMM.
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
