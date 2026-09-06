/// @file kernel/complex.hpp
/// @brief Raw-pointer kernels over complex scalars: mixed real/complex products
/// and the shifted-Hessenberg resolvent factorization.
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
/// Split out of `dense.hpp`/`factor.hpp` because `<complex>` is by far the
/// heaviest thing the kernel tier would otherwise pull: on libc++ it costs
/// ~95k preprocessed lines on its own — more than doubling the rest of the
/// tier combined — and drags the whole iostream/locale machinery with it, which
/// a freestanding or embedded target may not even have. Real-valued work (the
/// common case for a vendored kernel) now pays none of that; complex callers
/// include this header and opt in. `kernel/kernel.hpp` includes it, so nothing
/// changes for callers who take the umbrella.
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

// Shifted Hessenberg resolvent

/// @brief Factor \f$sI - H\f$ in place for an upper Hessenberg \f$H\f$.
///
/// Gaussian elimination needs to clear only one subdiagonal entry per column, so
/// this costs \f$O(n^2)\f$ rather than the \f$O(n^3)\f$ of a general LU. That is
/// what makes a Krylov resolvent affordable: the Hessenberg form is computed once
/// and each shift factors cheaply on top of it.
///
/// Partial pivoting compares the diagonal against the single subdiagonal entry,
/// since no other entry in the column can be larger.
///
/// @param work  In/out, n*n. Receives \f$sI - H\f$ and its factors.
/// @param H     Upper Hessenberg matrix, n*n row-major, real.
/// @param shift Complex shift \f$s\f$.
/// @param n     Dimension.
/// @param piv   Output pivot record, length n.
template <std::floating_point T, class Index>
inline void hessenberg_shifted_factor(std::complex<T> *NUM_K_RESTRICT work, const T *NUM_K_RESTRICT H,
                                      std::complex<T> shift, idx n,
                                      Index *NUM_K_RESTRICT piv) noexcept {
    using C = std::complex<T>;
    const T tiny = T(1e-30);

    for (idx i = 0; i < n; ++i) {
        const T *h_row = H + (i * n);
        C *m_row = work + (i * n);
        for (idx j = 0; j < n; ++j) {
            m_row[j] = (i == j ? shift : C(0, 0)) - h_row[j];
        }
    }

    for (idx i = 0; i + 1 < n; ++i) {
        C *row_i = work + (i * n);
        C *row_next = work + ((i + 1) * n);

        if (std::abs(row_next[i]) > std::abs(row_i[i])) {
            for (idx j = i; j < n; ++j) {
                std::swap(row_i[j], row_next[j]);
            }
            piv[i] = static_cast<Index>(i + 1);
        } else {
            piv[i] = static_cast<Index>(i);
        }

        const C pivot = row_i[i];
        if (std::abs(pivot) > tiny) {
            const C mult = row_next[i] / pivot;
            row_next[i] = mult;
            for (idx j = i + 1; j < n; ++j) {
                row_next[j] -= mult * row_i[j];
            }
        }
    }
}

/// @brief Substitute a right-hand side through a factored shifted Hessenberg system.
///
/// Separate from the factorization so that many right-hand sides share one
/// factorization at the same shift. `y` and `b` may alias.
template <std::floating_point T, class Index>
inline void hessenberg_shifted_substitute(std::complex<T> *y, const std::complex<T> *NUM_K_RESTRICT work,
                                          const Index *NUM_K_RESTRICT piv, const std::complex<T> *b,
                                          idx n) noexcept {
    using C = std::complex<T>;
    const T tiny = T(1e-30);

    for (idx i = 0; i < n; ++i) {
        y[i] = b[i];
    }
    for (idx i = 0; i + 1 < n; ++i) {
        if (static_cast<idx>(piv[i]) != i) {
            std::swap(y[i], y[i + 1]);
        }
        y[i + 1] -= work[((i + 1) * n) + i] * y[i];
    }
    for (idx step = 0; step < n; ++step) {
        const idx i = n - 1 - step;
        const C *row_i = work + (i * n);
        C sum = y[i];
        for (idx j = i + 1; j < n; ++j) {
            sum -= row_i[j] * y[j];
        }
        const C diag = row_i[i];
        y[i] = std::abs(diag) < tiny ? C(0, 0) : sum / diag;
    }
}

/// @brief Solve \f$(sI - H)\,y = b\f$ for a single right-hand side.
template <std::floating_point T, class Index>
inline void hessenberg_shifted_solve(std::complex<T> *y, const T *H, std::complex<T> shift,
                                     const std::complex<T> *b, idx n,
                                     std::complex<T> *NUM_K_RESTRICT work,
                                     Index *NUM_K_RESTRICT piv) noexcept {
    hessenberg_shifted_factor(work, H, shift, n, piv);
    hessenberg_shifted_substitute(y, work, piv, b, n);
}

} // namespace num::kernel
