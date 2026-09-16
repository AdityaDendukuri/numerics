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

/// @brief QR factorization \f$A=QR\f$.
struct qr_result {
    mat Q; ///< Orthonormal factor.
    mat R; ///< Upper-triangular factor.

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
qr_result qr(const mat &A);

/// @brief Solve \f$\min_x \|Ax-b\|_2\f$.
void qr_solve(const qr_result &f, const vec &b, vec &x);

namespace seq {
/// Blocked Householder QR through `kernel::qr_factor_blocked`, with Q formed
/// by applying the same compact-WY blocks, last to first, to the identity.
inline qr_result qr(const mat &A) {
    const idx m = A.rows();
    const idx n = A.cols();
    const idx r = std::min(m, n);
    const idx nb = std::min(kernel::qr_block, r);

    mat R = A;
    // Sized for the wider of the two block applications: the factorization's
    // trailing columns (up to n) and Q's trailing columns (up to m).
    array<real> tau(r), work(kernel::qr_workspace(m, std::max(m, n)));
    kernel::qr_factor_blocked(R.data(), n, m, n, tau.data(), work.data());

    // Q = H_0 ... H_{r-1}: for each block from the last, Q[k0:, k0:] <- B_k Q[k0:, k0:].
    // Entries of Q left of column k0 in those rows are still zero, so they
    // are skipped rather than multiplied.
    mat Q(m, m, real(0));
    for (idx i = 0; i < m; ++i) {
        Q(i, i) = real(1);
    }
    real *V = work.data();
    real *T = V + (m * nb);
    real *W = T + (nb * nb);
    idx k0 = r == 0 ? 0 : ((r - 1) / nb) * nb;
    for (idx blocks = (r + nb - 1) / nb; blocks-- > 0; k0 -= nb) {
        const idx kb = std::min(nb, r - k0);
        kernel::qr_form_block(V, T, R.data(), n, m, k0, kb, tau.data());
        kernel::qr_apply_block_left(&Q(k0, k0), m, m - k0, m - k0, V, T, kb, false, W);
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
inline qr_result qr(const mat &A) {
#if defined(NUMERICS_HAS_LAPACK)
    const idx m = A.rows(), n = A.cols();
    const idx k = std::min(m, n);

    mat R = A;
    array<double> tau(k);

    int info =
        LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, static_cast<lapack_int>(m), static_cast<lapack_int>(n),
                       R.data(), static_cast<lapack_int>(n), tau.data());
    if (info != 0) {
        throw std::runtime_error("qr (lapack): dgeqrf failed, info=" + std::to_string(info));
    }

    mat Rmat = R;
    for (idx i = 1; i < m; ++i) {
        for (idx j = 0; j < std::min(i, n); ++j) {
            Rmat(i, j) = 0.0;
        }
    }

    mat Q(m, m, 0.0);
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

inline qr_result qr(const mat &A) {
#if defined(NUMERICS_LAPACK_DEFAULT)
    return lapack::qr(A);
#else
    return seq::qr(A);
#endif
}

inline void qr_solve(const qr_result &f, const vec &b, vec &x) {
    const idx m = f.Q.rows();
    const idx n = f.R.cols();

    vec y(m, real(0));
    for (idx i = 0; i < m; ++i) {
        for (idx j = 0; j < m; ++j) {
            y[i] += f.Q(j, i) * b[j];
        }
    }

    vec xv(n, real(0));
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
