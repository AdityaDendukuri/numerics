/// @file qr.hpp
/// @brief QR factorization via Householder reflections.
#pragma once

#include "container/matrix.hpp"
#include "core/policy.hpp"
#include "core/types.hpp"
#include "kernel/kernel.hpp"
#include "lapack/lapack_wrapper.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <ostream>

namespace num {

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

/// @brief Apply \f$I - \beta v v^T\f$ from the left to an \f$m \times n\f$ block of `A`.
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
    kernel::gemm_transpose_left(W, cols, V, kb, C, ldc, T(1), T(0), rows, kb, cols);
    // W <- T' W, in place: T is upper triangular, so rows are consumed in the
    // order that keeps the unread ones intact.
    if (transpose) {
        for (idx i = kb; i-- > 0;) {
            T *NUM_K_RESTRICT w_i = W + (i * cols);
            kernel::scale(w_i, Tm[(i * kb) + i], cols);
            for (idx k = 0; k < i; ++k) {
                kernel::axpy(w_i, W + (k * cols), Tm[(k * kb) + i], cols);
            }
        }
    } else {
        for (idx i = 0; i < kb; ++i) {
            T *NUM_K_RESTRICT w_i = W + (i * cols);
            kernel::scale(w_i, Tm[(i * kb) + i], cols);
            for (idx k = i + 1; k < kb; ++k) {
                kernel::axpy(w_i, W + (k * cols), Tm[(i * kb) + k], cols);
            }
        }
    }
    // C <- C - V W.
    kernel::gemm(C, ldc, V, kb, W, cols, T(-1), T(1), rows, cols, kb);
}

/// @brief Compact Householder QR factorization; reflector tails remain below R's diagonal.
///
/// Each panel of `qr_block` columns is aggregated into a compact-WY block and applied to the
/// trailing columns with two `gemm`s. `tau` receives `min(m, n)` scalars; `work` holds
/// `qr_workspace(m, n)` elements.
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

} // namespace num

namespace num {

/// @brief QR factorization \f$A=QR\f$.
struct qr_result {
    mat<real> Q; ///< Orthonormal factor.
    mat<real> R; ///< Upper-triangular factor.

    friend std::ostream &operator<<(std::ostream &os, const qr_result &r) {
        os << "qr_result{ Q: " << r.Q.rows() << "x" << r.Q.cols() << ", R: " << r.R.rows() << "x"
           << r.R.cols() << " }";
        return os;
    }
};

/// @brief Factor \f$A\in\mathbb{R}^{m\times n}\f$ as \f$A=QR\f$.
///
/// Picks LAPACK (`dgeqrf`/`dorgqr`) if configured, else the in-tree blocked
/// Householder kernel. To force one explicitly, call `num::lapack::qr`/`num::seq::qr`.
/// @return `qr_result`: `.Q` (orthogonal), `.R` (upper triangular), with `A = Q*R`.
qr_result qr(const mat<real> &A);

/// @brief Solve \f$\min_x \|Ax-b\|_2\f$, which is \f$Ax = b\f$ for square nonsingular \f$A\f$.
/// `x` may be `b`.
void solve(const qr_result &f, const vec<real> &b, vec<real> &x);

namespace seq {
/// Blocked Householder QR through `num::qr_factor_blocked`, with Q formed
/// by applying the same compact-WY blocks, last to first, to the identity.
inline qr_result qr(const mat<real> &A) {
    const idx m = A.rows();
    const idx n = A.cols();
    const idx r = std::min(m, n);
    const idx nb = std::min(num::qr_block, r);

    mat<real> R = A;
    // Sized for the wider of the two block applications: the factorization's
    // trailing columns (up to n) and Q's trailing columns (up to m).
    array<real> tau(r), work(num::qr_workspace(m, std::max(m, n)));
    num::qr_factor_blocked(R.data(), n, m, n, tau.data(), work.data());

    // Q = H_0 ... H_{r-1}: for each block from the last, Q[k0:, k0:] <- B_k Q[k0:, k0:].
    // Entries of Q left of column k0 in those rows are still zero, so they
    // are skipped rather than multiplied.
    mat<real> Q(m, m, real(0));
    for (idx i = 0; i < m; ++i) {
        Q(i, i) = real(1);
    }
    real *V = work.data();
    real *T = V + (m * nb);
    real *W = T + (nb * nb);
    idx k0 = r == 0 ? 0 : ((r - 1) / nb) * nb;
    for (idx blocks = (r + nb - 1) / nb; blocks-- > 0; k0 -= nb) {
        const idx kb = std::min(nb, r - k0);
        num::qr_form_block(V, T, R.data(), n, m, k0, kb, tau.data());
        num::qr_apply_block_left(&Q(k0, k0), m, m - k0, m - k0, V, T, kb, false, W);
        if (k0 == 0) {
            break;
        }
    }

    for (idx i = 1; i < m; ++i) {
        for (idx j = 0; j < std::min(i, n); ++j) {
            R(i, j) = real(0);
        }
    }
    return {std::move(Q), std::move(R)};
}
} // namespace seq

namespace lapack {
inline qr_result qr(const mat<real> &A) {
#if defined(NUMERICS_HAS_LAPACK)
    const idx m = A.rows(), n = A.cols();
    const idx k = std::min(m, n);

    mat<real> R = A;
    array<double> tau(k);

    int info =
        LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, static_cast<lapack_int>(m), static_cast<lapack_int>(n),
                       R.data(), static_cast<lapack_int>(n), tau.data());
    if (info != 0) {
        throw std::runtime_error("qr (lapack): dgeqrf failed, info=" + std::to_string(info));
    }

    mat<real> Rmat = R;
    for (idx i = 1; i < m; ++i) {
        for (idx j = 0; j < std::min(i, n); ++j) {
            Rmat(i, j) = 0.0;
        }
    }

    mat<real> Q(m, m, 0.0);
    for (idx j = 0; j < k; ++j) {
        for (idx i = 0; i < m; ++i) {
            Q(i, j) = R(i, j);
        }
    }

    info = LAPACKE_dorgqr(LAPACK_ROW_MAJOR, static_cast<lapack_int>(m), static_cast<lapack_int>(m),
                          static_cast<lapack_int>(k), Q.data(), static_cast<lapack_int>(m),
                          tau.data());
    if (info != 0) {
        throw std::runtime_error("qr (lapack): dorgqr failed, info=" + std::to_string(info));
    }

    return {std::move(Q), std::move(Rmat)};
#else
    return seq::qr(A);
#endif
}
} // namespace lapack

/// The blocked kernel QR; it measured faster than dgeqrf + dorgqr through
/// LAPACKE at every size tried, and `num::lapack::qr` remains callable by name.
inline qr_result qr(const mat<real> &A) {
    return seq::qr(A);
}

inline void solve(const qr_result &f, const vec<real> &b, vec<real> &x) {
    const idx m = f.Q.rows();
    const idx n = f.R.cols();

    vec<real> y(m, real(0));
    for (idx i = 0; i < m; ++i) {
        for (idx j = 0; j < m; ++j) {
            y[i] += f.Q(j, i) * b[j];
        }
    }

    vec<real> xv(n, real(0));
    for (idx i = n; i-- > 0;) {
        xv[i] = y[i];
        for (idx j = i + 1; j < n; ++j) {
            xv[i] -= f.R(i, j) * xv[j];
        }
        xv[i] /= f.R(i, i);
    }

    x = std::move(xv);
}

} // namespace num
