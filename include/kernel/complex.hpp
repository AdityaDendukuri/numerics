/// @file kernel/complex.hpp
/// @brief Mixed real/complex kernels over raw pointers.
///
/// SPDX-License-Identifier: MIT
/// Part of numerics, (c) 2026 Aditya Dendukuri.
/// https://github.com/AdityaDendukuri/numerics
///
/// Separate from the real kernels because `<complex>` is expensive to include.
#pragma once

#include "kernel/vector.hpp"
#include <algorithm>
#include <complex>
#include <concepts>

namespace num::kernel {

/// @brief Real matrix, complex vector product: \f$x \leftarrow Q y\f$.
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

/// @brief Real transposed matrix-vector product, complex result: \f$x \leftarrow Q^T y\f$.
/// `y` may be real or complex.
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

} // namespace num::kernel
