/// @file kernel/rotations.hpp
/// @brief Raw-pointer kernels: Givens, Householder, and Jacobi rotations; blocked QR.
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

#include "kernel/dense.hpp"
#include "kernel/vector.hpp"
#include <algorithm>
#include <cmath>
#include <concepts>

namespace num::kernel {

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

// Householder Reflections & Blocked QR

/// @brief Computes elementary Householder reflector vector \f$\mathbf{v}\f$ and scalar \f$\beta\f$
/// such that:
/// \f[
/// (I - \beta \mathbf{v} \mathbf{v}^T) \mathbf{x} = \mp \|\mathbf{x}\|_2 \mathbf{e}_1
/// \f]
/// \f$\mathbf{v}\f$ is sized \f$m\f$, with \f$v_0 = 1\f$ implicitly assigned.
template <std::floating_point T>
NUM_K_AINLINE void householder_vector(T *NUM_K_RESTRICT v, T &beta, const T *NUM_K_RESTRICT x,
                                      idx m) noexcept {
    // LAPACK's convention (dlarfg): a column whose tail is already zero needs
    // no reflection, so beta = 0 and the column is left as it is. Without this
    // a length-one column gets H = -1, which flips a sign in R that a Q built
    // from the other reflectors never sees.
    T tail = T(0);
    NUM_K_IVDEP
    for (idx i = 1; i < m; ++i) {
        tail += x[i] * x[i];
    }
    const T norm_x = std::sqrt(tail + (x[0] * x[0]));
    if (tail == T(0) || norm_x < T(1e-15)) {
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

/// @brief Householder reflector for a strided column of a matrix.
///
/// The column is copied into `v` and the reflector formed there in place. Not
/// a call to `householder_vector(v, beta, v, m)`: that passes one buffer as
/// two `restrict` parameters, and at `-O3` the compiler is entitled to read
/// `x[0]` after `v[0]` has been overwritten with 1.
template <std::floating_point T>
NUM_K_AINLINE void householder_vector_strided(T *NUM_K_RESTRICT v, T &beta,
                                              const T *NUM_K_RESTRICT A, idx lda, idx offset,
                                              idx m) noexcept {
    T tail = T(0);
    v[0] = A[(offset * lda) + offset];
    for (idx i = 1; i < m; ++i) {
        v[i] = A[((offset + i) * lda) + offset];
        tail += v[i] * v[i];
    }
    const T norm_x = std::sqrt(tail + (v[0] * v[0]));
    if (tail == T(0) || norm_x < T(1e-15)) { // see householder_vector
        beta = T(0);
        v[0] = T(1);
        return;
    }
    const T sign = (v[0] >= T(0)) ? T(1) : T(-1);
    const T mu = v[0] + (sign * norm_x);
    v[0] = T(1);
    T v_sq = T(1);
    for (idx i = 1; i < m; ++i) {
        v[i] /= mu;
        v_sq += v[i] * v[i];
    }
    beta = T(2) / v_sq;
}

template <std::floating_point T>
NUM_K_AINLINE void householder_left(T *NUM_K_RESTRICT A, idx lda, const T *NUM_K_RESTRICT v, T beta,
                                    idx m, idx n, T *NUM_K_RESTRICT work) noexcept;

/// @brief Reflectors per block in `qr_factor_blocked`.
inline constexpr idx qr_block = 32;

/// @brief Workspace elements `qr_factor_blocked` needs for an `m x n` factorization.
[[nodiscard]] constexpr idx qr_workspace(idx m, idx n) noexcept {
    const idx nb = std::min(qr_block, std::min(m, n));
    return (m * nb) + (nb * nb) + (nb * n) + m + n; // V, T, W, v, row work
}

/// @brief Form the compact-WY block reflector for columns `[k0, k0 + kb)`.
///
/// Reads the reflector tails stored below the diagonal of the factored panel
/// and their scalars `tau`, and writes `V` (`(m - k0) x kb`, unit lower
/// trapezoidal, row stride `kb`) and the upper triangular `T` (`kb x kb`)
/// with \f$H_{k_0} \cdots H_{k_0 + k_b - 1} = I - V T V^T\f$.
template <std::floating_point T>
inline void qr_form_block(T *NUM_K_RESTRICT V, T *NUM_K_RESTRICT Tm, const T *NUM_K_RESTRICT A,
                          idx lda, idx m, idx k0, idx kb, const T *NUM_K_RESTRICT tau) noexcept {
    const idx rows = m - k0;
    for (idx i = 0; i < rows; ++i) {
        for (idx j = 0; j < kb; ++j) {
            V[(i * kb) + j] = i < j ? T(0) : (i == j ? T(1) : A[((k0 + i) * lda) + k0 + j]);
        }
    }
    // T(0:j, j) = -tau_j * T(0:j, 0:j) * w with w = V(:, 0:j)^T V(:, j); T(j, j) = tau_j.
    // Column j of T holds w while it is being consumed: entry i of the product
    // reads T(i, i:j) from finished columns and w_k = T(k, j) for k >= i, none
    // of which has been overwritten yet when row i is written.
    for (idx j = 0; j < kb; ++j) {
        const T tau_j = tau[k0 + j];
        for (idx i = 0; i < j; ++i) {
            T w = T(0);
            for (idx r = j; r < rows; ++r) {
                w += V[(r * kb) + i] * V[(r * kb) + j];
            }
            Tm[(i * kb) + j] = w;
        }
        for (idx i = 0; i < j; ++i) {
            T sum = T(0);
            for (idx k = i; k < j; ++k) {
                sum += Tm[(i * kb) + k] * Tm[(k * kb) + j];
            }
            Tm[(i * kb) + j] = -tau_j * sum;
        }
        for (idx i = j + 1; i < kb; ++i) {
            Tm[(i * kb) + j] = T(0);
        }
        Tm[(j * kb) + j] = tau_j;
    }
}

/// @brief `C <- (I - V T' V^T) C` for a compact-WY block reflector.
///
/// `C` is `rows x cols`; `V` and `T` come from `qr_form_block` with `kb`
/// reflectors; `T'` is `T^T` when `transpose` is set (the product
/// \f$H_{k_b-1} \cdots H_0\f$, used when factoring) and `T` otherwise
/// (\f$H_0 \cdots H_{k_b-1}\f$, used when forming Q). `W` holds `kb x cols`.
template <std::floating_point T>
inline void qr_apply_block_left(T *NUM_K_RESTRICT C, idx ldc, idx rows, idx cols,
                                const T *NUM_K_RESTRICT V, const T *NUM_K_RESTRICT Tm, idx kb,
                                bool transpose, T *NUM_K_RESTRICT W) noexcept {
    // W <- V^T C.
    gemm_transpose_left(W, cols, V, kb, C, ldc, T(1), T(0), rows, kb, cols);
    // W <- T' W, in place: T is upper triangular, so rows are consumed in the
    // order that keeps the unread ones intact.
    if (transpose) {
        for (idx i = kb; i-- > 0;) {
            T *NUM_K_RESTRICT w_i = W + (i * cols);
            scale(w_i, Tm[(i * kb) + i], cols);
            for (idx k = 0; k < i; ++k) {
                axpy(w_i, W + (k * cols), Tm[(k * kb) + i], cols);
            }
        }
    } else {
        for (idx i = 0; i < kb; ++i) {
            T *NUM_K_RESTRICT w_i = W + (i * cols);
            scale(w_i, Tm[(i * kb) + i], cols);
            for (idx k = i + 1; k < kb; ++k) {
                axpy(w_i, W + (k * cols), Tm[(i * kb) + k], cols);
            }
        }
    }
    // C <- C - V W.
    gemm(C, ldc, V, kb, W, cols, T(-1), T(1), rows, cols, kb);
}

/// @brief Compact Householder QR factorization; reflector tails remain below R's diagonal.
///
/// Blocked: each panel of `qr_block` columns is factored by unblocked
/// Householder reflections applied only within the panel, then the panel's
/// reflectors are aggregated into one compact-WY block (`qr_form_block`) and
/// applied to the trailing columns with two `gemm`s. `tau` receives
/// `min(m, n)` scalars; `work` holds `qr_workspace(m, n)` elements.
template <std::floating_point T>
inline void qr_factor_blocked(T *NUM_K_RESTRICT A, idx lda, idx m, idx n, T *NUM_K_RESTRICT tau,
                              T *NUM_K_RESTRICT work) noexcept {
    const idx r = std::min(m, n);
    const idx nb = std::min(qr_block, r);
    T *NUM_K_RESTRICT V = work;
    T *NUM_K_RESTRICT Tm = V + (m * nb);
    T *NUM_K_RESTRICT W = Tm + (nb * nb);
    T *NUM_K_RESTRICT v = W + (nb * n);
    T *NUM_K_RESTRICT row_work = v + m;

    for (idx k0 = 0; k0 < r; k0 += nb) {
        const idx kb = std::min(nb, r - k0);
        const idx panel_end = k0 + kb;
        for (idx k = k0; k < panel_end; ++k) {
            const idx len = m - k;
            T beta = T(0);
            householder_vector_strided(v, beta, A, lda, k, len);
            tau[k] = beta;
            if (beta != T(0)) {
                // Panel columns only; the trailing block gets the aggregate below.
                householder_left(A + (k * lda) + k, lda, v, beta, len, panel_end - k, row_work);
            }
            for (idx i = 1; i < len; ++i) {
                A[((k + i) * lda) + k] = v[i];
            }
        }
        if (panel_end < n) {
            qr_form_block(V, Tm, A, lda, m, k0, kb, tau);
            qr_apply_block_left(A + (k0 * lda) + panel_end, lda, m - k0, n - panel_end, V, Tm, kb,
                                true, W);
        }
    }
}

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

// Jacobi Rotations

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

} // namespace num::kernel
