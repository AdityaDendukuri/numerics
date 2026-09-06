/// @file qr.hpp
/// @brief QR factorization via Householder reflections.
#pragma once

#include "core/types.hpp"
#include "kernel/kernel.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include "lapack/lapack_wrapper.hpp"
#include "container/matrix.hpp"
#include "core/policy.hpp"

#include <ostream>

namespace num {

/// @brief QR factorization \f$A=QR\f$.
struct qr_result {
    mat Q; ///< Orthonormal factor.
    mat R; ///< Upper-triangular factor.

    friend std::ostream &operator<<(std::ostream &os, const qr_result &r) {
        os << "qr_result{ Q: " << r.Q.rows() << "x" << r.Q.cols()
           << ", R: " << r.R.rows() << "x" << r.R.cols() << " }";
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
inline qr_result qr(const mat &A) {
    const idx m = A.rows();
    const idx n = A.cols();
    const idx r = (m > n) ? n : m - 1;

    mat R = A;
    array<array<real>> vs(r);
    array<real> betas(r, 0.0);
    array<real> tau(r), v(m), work(n);
    // A <- compact Householder QR; reflector tails remain below R's diagonal.
    kernel::qr_factor_blocked(R.data(), n, m, n, tau.data(), v.data(), work.data());
    for (idx k = 0; k < r; ++k) {
        const idx len = m - k;
        betas[k] = tau[k];
        vs[k].assign(len, real(0));
        vs[k][0] = real(1);
        for (idx i = 1; i < len; ++i) vs[k][i] = R((k + i), k);
    }

    mat Q(m, m, real(0));
    for (idx i = 0; i < m; ++i) {
        Q(i, i) = real(1);
    }

    for (idx k = r; k-- > 0;) {
        if (betas[k] == 0.0) {
            continue;
        }
        const array<real> &v = vs[k];
        const idx len = static_cast<idx>(v.size());
        array<real> work(m - k);
        kernel::householder_left(&Q(k, k), m, v.data(), betas[k], len, m - k, work.data());
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
#if defined(NUMERICS_HAS_LAPACK)
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
