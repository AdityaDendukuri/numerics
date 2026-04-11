/// @file svd/svd.cpp
/// @brief SVD dispatcher + randomized truncated SVD.
///
/// One-sided Jacobi SVD
///
/// Apply Givens rotations G to the columns of A from the right until the
/// columns are mutually orthogonal.  Each rotation zeros (A^T*A)[p,q]:
///
///   Given col_p and col_q of the working matrix A:
///   alpha = col_p * col_p,  beta = col_q * col_q,  gamma = col_p * col_q
///   zeta = (beta - alpha) / (2gamma)
///   t = sign(zeta) / (|zeta| + sqrt(1 + zeta^2))
///   c = 1/sqrt(1+t^2),  s = c*t
///
/// Randomized SVD
///
/// Halko, Martinsson, Tropp (2011) "Finding Structure with Randomness".
///
/// Backend routing:
///   Backend::lapack  -> backends::lapack::svd  (LAPACKE_dgesdd,
///   divide-and-conquer) everything else  -> backends::seq::svd     (one-sided
///   Jacobi)

#include "linalg/svd/svd.hpp"
#include "linalg/factorization/qr.hpp"
#include "backends/seq/impl.hpp"
#include "backends/lapack/impl.hpp"

namespace num {

SVDResult svd(const Matrix& A_in, Backend backend, real tol, idx max_sweeps) {
    switch (backend) {
        case Backend::lapack:
            return backends::lapack::svd(A_in);
        default:
            return backends::seq::svd(A_in, tol, max_sweeps);
    }
}

// Randomized truncated SVD  -- Halko, Martinsson, Tropp (2011)
//
// Steps:
//   1. Draw Omega in R^{nx(k+p)}  (Gaussian random sketch)
//   2. Y = A * Omega
//   3. Q, _ = QR(Y)            (orthonormal basis for approx. range of A)
//   4. B = Q^T * A              (project to small k-dimensional subspace)
//   5. U~, Sigma, V^T = svd(B)      (cheap: B is (k+p)xn, small)
//   6. U = Q * U~               (lift back to full dimension)

SVDResult svd_truncated(const Matrix& A,
                        idx           k,
                        Backend       backend,
                        idx           oversampling,
                        Rng*          rng) {
    const idx m = A.rows(), n = A.cols();
    if (k == 0 || k > std::min(m, n))
        throw std::invalid_argument("svd_truncated: k out of range");

    const idx l = k + oversampling;

    Rng local_rng;
    if (!rng)
        rng = &local_rng;

    // 1. Gaussian sketch matrix Omega in R^{nxl}
    Matrix Omega(n, l);
    for (idx j = 0; j < l; ++j)
        for (idx i = 0; i < n; ++i)
            Omega(i, j) = rng_normal(rng, 0.0, 1.0);

    // 2. Y = A * Omega  (m x l)
    Matrix Y(m, l, 0.0);
    matmul(A, Omega, Y, backend);

    // 3. Q = QR(Y).Q  (m x m, orthonormal columns)
    QRResult      qr_res = qr(Y);
    const Matrix& Q      = qr_res.Q;

    // 4. B = Q^T * A  (l x n)
    Matrix B(l, n, 0.0);
    for (idx i = 0; i < l; ++i)
        for (idx kk = 0; kk < m; ++kk) {
            const real q_ki = Q(kk, i);
            for (idx j = 0; j < n; ++j)
                B(i, j) += q_ki * A(kk, j);
        }

    // 5. SVD of B (small: l x n)
    SVDResult small = svd(B, backend);

    // 6. U = Q * U~[:, 0..k-1]  (m x k)
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
