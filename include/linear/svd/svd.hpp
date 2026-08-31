/// @file linear/svd/svd.hpp
/// @brief Dense and randomized truncated SVD.
#pragma once

#include "container/matrix.hpp"
#include "container/matrix_ops.hpp"
#include "container/parallel/lapack_wrapper.hpp"
#include "container/util/math.hpp"
#include "container/vector.hpp"
#include "core/policy.hpp"
#include "linear/factorization/qr.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <ostream>
#include <string>

namespace num {

/// Singular value decomposition and convergence metadata.
struct SVDResult {
    Matrix U;               ///< Left singular vectors.
    Vector S;               ///< Singular values in descending order.
    Matrix Vt;              ///< Transposed right singular vectors.
    idx sweeps = 0;         ///< Jacobi sweeps for the fallback implementation.
    bool converged = false; ///< Whether the requested tolerance was met.

    friend std::ostream &operator<<(std::ostream &os, const SVDResult &r) {
        os << "SVDResult{ rank: " << r.S.size()
           << ", converged: " << (r.converged ? "true" : "false")
           << ", sweeps: " << r.sweeps
           << ", U: " << r.U.rows() << "x" << r.U.cols()
           << ", Vt: " << r.Vt.rows() << "x" << r.Vt.cols() << " }";
        return os;
    }
};

/// @brief Compute full singular value decomposition \f$A = U \Sigma V^T\f$.
///
/// Dispatches to LAPACK divide-and-conquer (`dgesdd`) when available, or executes
/// in-tree one-sided Hestenes-Jacobi orthogonalization sweeps with Givens rotations.
///
/// @param A Input \f$m \times n\f$ dense matrix.
/// @param backend Execution backend tag (`backend::factor`, `backend::lapack`, `backend::seq`).
/// @param tol Orthogonality tolerance for Jacobi sweeps (default: 1e-12).
/// @param max_sweeps Maximum one-sided Jacobi sweeps (default: 100).
/// @return `SVDResult` with left singular vectors \f$U\f$, singular values \f$\Sigma\f$, and transposed right vectors \f$V^T\f$.
/// @see svd_truncated, eig_sym, qr
SVDResult svd(const Matrix &A, Backend backend = backend::factor, real tol = 1e-12,
              idx max_sweeps = 100);

/// @brief Compute randomized truncated rank-\f$k\f$ SVD approximation \f$A \approx U_k \Sigma_k V_k^T\f$.
///
/// Uses Gaussian random test matrices and QR range-finder to project \f$A\f$ into a small
/// subspace of dimension \f$l = k + \text{oversampling}\f$, achieving near-optimal low-rank reconstruction.
///
/// @param A Input \f$m \times n\f$ dense matrix.
/// @param k Target low-rank approximation dimension (\f$0 < k \le \min(m, n)\f$).
/// @param backend Matrix multiplication backend.
/// @param oversampling Additional random test vectors for spectral gap safety (default: 10).
/// @param rng Optional pointer to custom random number generator for reproducible sampling.
/// @return `SVDResult` containing rank-\f$k\f$ truncated factors \f$U_k, \Sigma_k, V_k^T\f$.
/// @throws std::invalid_argument If \f$k\f$ is out of range.
/// @see svd, lanczos
SVDResult svd_truncated(const Matrix &A, idx k, Backend backend = backend::dflt,
                        idx oversampling = 10, Rng *rng = nullptr);

namespace backends {

namespace seq {
inline SVDResult svd(const Matrix &A_in, real tol, idx max_sweeps) {
    constexpr real tiny = 1e-300;
    idx m = A_in.rows(), n = A_in.cols();
    idx r = std::min(m, n);

    Matrix A = A_in;
    Matrix V(n, n, 0.0);
    for (idx i = 0; i < n; ++i) {
        V(i, i) = 1.0;
    }

    idx sweeps = 0;
    bool converged = false;

    for (idx sweep = 0; sweep < max_sweeps; ++sweep) {
        real max_cos = 0;
        for (idx p = 0; p < r - 1; ++p) {
            for (idx q = p + 1; q < r; ++q) {
                const real alpha = kernel::raw::column_dot(A.data(), n, m, p, p);
                const real beta = kernel::raw::column_dot(A.data(), n, m, q, q);
                const real gamma = kernel::raw::column_dot(A.data(), n, m, p, q);
                if (alpha < tiny || beta < tiny) {
                    continue;
                }

                real cos_pq = std::abs(gamma) / std::sqrt(alpha * beta);
                max_cos = std::max(max_cos, cos_pq);

                if (cos_pq < tol) {
                    continue;
                }

                real zeta = (beta - alpha) / (2.0 * gamma);
                real t =
                    std::copysign(1.0, zeta) / (std::abs(zeta) + std::sqrt(1.0 + (zeta * zeta)));
                real c = 1.0 / std::sqrt(1.0 + (t * t));
                real s = c * t;

                // [A_p A_q] <- [A_p A_q] J(c,s).
                kernel::raw::rotate_columns(A.data(), n, m, p, q, c, s);
                kernel::raw::rotate_columns(V.data(), n, n, p, q, c, s);
            }
        }

        ++sweeps;
        if (max_cos < tol) {
            converged = true;
            break;
        }
    }

    Vector S(r);
    Matrix U(m, r, 0.0);
    for (idx j = 0; j < r; ++j) {
        const real nrm = kernel::raw::norm_sq_strided(A.data() + j, n, m);
        S[j] = std::sqrt(nrm);
        if (S[j] > tiny) {
            // U[:,j] <- A[:,j] / sigma_j.
            kernel::raw::scale_copy_strided(U.data() + j, r, A.data() + j, n, real(1) / S[j], m);
        }
    }

    for (idx i = 0; i < r - 1; ++i) {
        idx max_j = i;
        for (idx j = i + 1; j < r; ++j) {
            if (S[j] > S[max_j]) {
                max_j = j;
            }
        }

        if (max_j != i) {
            std::swap(S[i], S[max_j]);
            // [U_i U_j] <- [U_j U_i], [V_i V_j] <- [V_j V_i].
            kernel::raw::swap_strided(U.data() + i, r, U.data() + max_j, r, m);
            kernel::raw::swap_strided(V.data() + i, n, V.data() + max_j, n, n);
        }
    }

    Matrix vt(r, n, 0.0);
    // V^T <- transpose(V[:,0:r]).
    for (idx i = 0; i < r; ++i) {
        kernel::raw::copy_strided(vt.data() + (i * n), 1, V.data() + i, n, n);
    }

    return {U, S, vt, sweeps, converged};
}
} // namespace seq

namespace lapack {
inline SVDResult svd(const Matrix &A_in) {
#if defined(NUMERICS_HAS_LAPACK)
    const idx m = A_in.rows(), n = A_in.cols();
    const idx r = std::min(m, n);
    Matrix Aw = A_in;
    Vector S(r);
    Matrix U(m, r);
    Matrix Vt(r, n);

    int info =
        LAPACKE_dgesdd(LAPACK_ROW_MAJOR, 'S', static_cast<lapack_int>(m),
                       static_cast<lapack_int>(n), Aw.data(), static_cast<lapack_int>(n), S.data(),
                       U.data(), static_cast<lapack_int>(r), Vt.data(), static_cast<lapack_int>(n));
    if (info != 0) {
        throw std::runtime_error("svd (lapack): dgesdd failed, info=" + std::to_string(info));
    }

    return {std::move(U), std::move(S), std::move(Vt), 0, true};
#else
    return seq::svd(A_in, 1e-12, 100);
#endif
}
} // namespace lapack

} // namespace backends

inline SVDResult svd(const Matrix &A_in, Backend backend, real tol, idx max_sweeps) {
    switch (backend) {
    case backend::lapack:
        return backends::lapack::svd(A_in);
    default:
        return backends::seq::svd(A_in, tol, max_sweeps);
    }
}

inline SVDResult svd_truncated(const Matrix &A, idx k, Backend backend, idx oversampling,
                               Rng *rng) {
    const idx m = A.rows(), n = A.cols();
    if (k == 0 || k > std::min(m, n)) {
        throw std::invalid_argument("svd_truncated: k out of range");
    }

    const idx l = k + oversampling;

    Rng local_rng;
    if (!rng) {
        rng = &local_rng;
    }

    Matrix Omega(n, l);
    for (idx j = 0; j < l; ++j) {
        for (idx i = 0; i < n; ++i) {
            Omega(i, j) = rng_normal(rng, 0.0, 1.0);
        }
    }

    Matrix Y(m, l, 0.0);
    matmul(A, Omega, Y, backend);

    QRResult qr_res = qr(Y);
    const Matrix &Q = qr_res.Q;

    Matrix B(l, n, 0.0);
    // B <- Q_l^T*A
    kernel::raw::gemm_transpose_left(B.data(), B.cols(), Q.data(), Q.cols(), A.data(), A.cols(),
                                     real(1), real(0), m, l, n);

    SVDResult small = svd(B, backend);

    Matrix U(m, k, 0.0);
    // U_k <- Q_l*U(B)[:,0:k]
    kernel::raw::gemm(U.data(), U.cols(), Q.data(), Q.cols(), small.U.data(), small.U.cols(),
                      real(1), real(0), m, k, l);

    Vector S(k);
    // sigma_k <- sigma(B)[0:k]
    kernel::raw::copy(S.data(), small.S.data(), k);

    Matrix Vt(k, n, 0.0);
    // V_k^T <- V(B)^T[0:k,:]
    kernel::raw::copy(Vt.data(), small.Vt.data(), k * n);

    return {U, S, Vt, 0, true};
}

} // namespace num
