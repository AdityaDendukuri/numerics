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

/// @brief Householder reflector for a strided column of a matrix.
template <std::floating_point T>
NUM_K_AINLINE void householder_vector_strided(T *NUM_K_RESTRICT v, T &beta,
                                              const T *NUM_K_RESTRICT A, idx lda, idx offset,
                                              idx m) noexcept {
    for (idx i = 0; i < m; ++i)
        v[i] = A[((offset + i) * lda) + offset];
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
        if (beta == T(0))
            continue;
        // R(k:m,k:n) <- H_k R(k:m,k:n).
        householder_left(A + (k * lda) + k, lda, v, beta, len, n - k, work);
        for (idx i = 1; i < len; ++i)
            A[((k + i) * lda) + k] = v[i];
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
