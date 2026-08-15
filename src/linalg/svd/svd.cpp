/// @file svd/svd.cpp
/// @brief SVD dispatcher + implementations (seq, lapack) + randomized truncated SVD.

#include "linalg/svd/svd.hpp"
#include "linalg/factorization/qr.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#if defined(NUMERICS_HAS_LAPACK)
    #include <lapacke.h>
#endif

namespace num {

namespace backends {

namespace seq {
SVDResult svd(const Matrix& A_in, real tol, idx max_sweeps) {
    constexpr real tiny = 1e-300;
    idx m = A_in.rows(), n = A_in.cols();
    idx r = std::min(m, n);

    Matrix A = A_in;
    Matrix V(n, n, 0.0);
    for (idx i = 0; i < n; ++i)
        V(i, i) = 1.0;

    idx sweeps = 0;
    bool converged = false;

    for (idx sweep = 0; sweep < max_sweeps; ++sweep) {
        real max_cos = 0;
        for (idx p = 0; p < r - 1; ++p) {
            for (idx q = p + 1; q < r; ++q) {
                real alpha = 0, beta = 0, gamma = 0;
                for (idx i = 0; i < m; ++i) {
                    alpha += A(i, p) * A(i, p);
                    beta += A(i, q) * A(i, q);
                    gamma += A(i, p) * A(i, q);
                }
                if (alpha < tiny || beta < tiny)
                    continue;

                real cos_pq = std::abs(gamma) / std::sqrt(alpha * beta);
                max_cos = std::max(max_cos, cos_pq);

                if (cos_pq < tol)
                    continue;

                real zeta = (beta - alpha) / (2.0 * gamma);
                real t = std::copysign(1.0, zeta) / (std::abs(zeta) + std::sqrt(1.0 + zeta * zeta));
                real c = 1.0 / std::sqrt(1.0 + t * t);
                real s = c * t;

                for (idx i = 0; i < m; ++i) {
                    real aip = A(i, p), aiq = A(i, q);
                    A(i, p) = c * aip - s * aiq;
                    A(i, q) = s * aip + c * aiq;
                }

                for (idx i = 0; i < n; ++i) {
                    real vip = V(i, p), viq = V(i, q);
                    V(i, p) = c * vip - s * viq;
                    V(i, q) = s * vip + c * viq;
                }
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
        real nrm = 0;
        for (idx i = 0; i < m; ++i)
            nrm += A(i, j) * A(i, j);
        S[j] = std::sqrt(nrm);
        if (S[j] > tiny)
            for (idx i = 0; i < m; ++i)
                U(i, j) = A(i, j) / S[j];
    }

    for (idx i = 0; i < r - 1; ++i) {
        idx max_j = i;
        for (idx j = i + 1; j < r; ++j)
            if (S[j] > S[max_j])
                max_j = j;

        if (max_j != i) {
            std::swap(S[i], S[max_j]);
            for (idx k = 0; k < m; ++k)
                std::swap(U(k, i), U(k, max_j));
            for (idx k = 0; k < n; ++k)
                std::swap(V(k, i), V(k, max_j));
        }
    }

    Matrix Vt(r, n, 0.0);
    for (idx i = 0; i < r; ++i)
        for (idx j = 0; j < n; ++j)
            Vt(i, j) = V(j, i);

    return {U, S, Vt, sweeps, converged};
}
} // namespace seq

namespace lapack {
SVDResult svd(const Matrix& A_in) {
#if defined(NUMERICS_HAS_LAPACK)
    const idx m = A_in.rows(), n = A_in.cols();
    const idx r = std::min(m, n);
    Matrix Aw = A_in;
    Vector S(r);
    Matrix U(m, r);
    Matrix Vt(r, n);

    int info = LAPACKE_dgesdd(LAPACK_ROW_MAJOR,
                              'S',
                              static_cast<lapack_int>(m),
                              static_cast<lapack_int>(n),
                              Aw.data(),
                              static_cast<lapack_int>(n),
                              S.data(),
                              U.data(),
                              static_cast<lapack_int>(r),
                              Vt.data(),
                              static_cast<lapack_int>(n));
    if (info != 0)
        throw std::runtime_error("svd (lapack): dgesdd failed, info=" + std::to_string(info));

    return {std::move(U), std::move(S), std::move(Vt), 0, true};
#else
    return seq::svd(A_in, 1e-12, 100);
#endif
}
} // namespace lapack

} // namespace backends

SVDResult svd(const Matrix& A_in, Backend backend, real tol, idx max_sweeps) {
    switch (backend) {
        case Backend::lapack:
            return backends::lapack::svd(A_in);
        default:
            return backends::seq::svd(A_in, tol, max_sweeps);
    }
}

SVDResult svd_truncated(const Matrix& A,
                        idx k,
                        Backend backend,
                        idx oversampling,
                        Rng* rng) {
    const idx m = A.rows(), n = A.cols();
    if (k == 0 || k > std::min(m, n))
        throw std::invalid_argument("svd_truncated: k out of range");

    const idx l = k + oversampling;

    Rng local_rng;
    if (!rng)
        rng = &local_rng;

    Matrix Omega(n, l);
    for (idx j = 0; j < l; ++j)
        for (idx i = 0; i < n; ++i)
            Omega(i, j) = rng_normal(rng, 0.0, 1.0);

    Matrix Y(m, l, 0.0);
    matmul(A, Omega, Y, backend);

    QRResult qr_res = qr(Y);
    const Matrix& Q = qr_res.Q;

    Matrix B(l, n, 0.0);
    for (idx i = 0; i < l; ++i)
        for (idx kk = 0; kk < m; ++kk) {
            const real q_ki = Q(kk, i);
            for (idx j = 0; j < n; ++j)
                B(i, j) += q_ki * A(kk, j);
        }

    SVDResult small = svd(B, backend);

    Matrix U(m, k, 0.0);
    for (idx j = 0; j < k; ++j)
        for (idx i = 0; i < m; ++i)
            for (idx ii = 0; ii < l; ++ii)
                U(i, j) += Q(i, ii) * small.U(ii, j);

    Vector S(k);
    for (idx i = 0; i < k; ++i)
        S[i] = small.S[i];

    Matrix Vt(k, n, 0.0);
    for (idx i = 0; i < k; ++i)
        for (idx j = 0; j < n; ++j)
            Vt(i, j) = small.Vt(i, j);

    return {U, S, Vt, 0, true};
}

} // namespace num
