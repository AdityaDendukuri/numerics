/// @file kernel/complex.hpp
/// @brief Raw-pointer kernels over complex scalars: mixed real/complex products.
///
/// SPDX-License-Identifier: MIT
/// Part of numerics, (c) 2026 Aditya Dendukuri.
/// https://github.com/AdityaDendukuri/numerics
///
/// Depends only on kernel/vector.hpp. Keep the two attribution lines above with whatever you
/// copy. Kept apart from the real kernels because `<complex>` costs about 95k preprocessed
/// lines on libc++; `kernel/kernel.hpp` still includes it.
#pragma once

#include "kernel/vector.hpp"
#include <algorithm>
#include <complex>
#include <concepts>

namespace num::kernel {

// Mixed real/complex products

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

/// @brief Mixed transpose product \f$x = Q^T y\f$ with a real matrix and complex result. The
/// input scalar is a separate parameter, so `y` may be real or complex.
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

} // namespace num::kernel
