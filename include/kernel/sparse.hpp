/// @file kernel/sparse.hpp
/// @brief Sparse kernels over raw pointers: CSR matrix-vector and matrix-matrix products.
///
/// SPDX-License-Identifier: MIT
/// Part of numerics, (c) 2026 Aditya Dendukuri.
/// https://github.com/AdityaDendukuri/numerics
#pragma once

#include "kernel/vector.hpp"
#include <cmath>
#include <concepts>
#include <type_traits>

namespace num::kernel {

namespace detail {

/// @brief Row length below which a plain sum is faster than `reduce`.
template <std::floating_point T>
inline constexpr idx short_row_cutoff = 4 * (NUM_K_VECTOR_BYTES / sizeof(T));

/// @brief Compute the dot product of one CSR row and a vector: \f$\sum_k a_{ik} x_k\f$.
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

/// @brief Multiply a CSR sparse matrix and a vector (SpMV): \f$y \leftarrow A x\f$.
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

/// @brief Multiply a CSR sparse matrix and a vector, scaled:
/// \f$y \leftarrow \alpha A x + \beta y\f$.
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

/// @brief Multiply a CSR sparse matrix and a dense matrix (SpMM): \f$Y \leftarrow A X\f$.
/// `X` and `Y` are row-major with `nrhs` columns.
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
