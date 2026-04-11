/// @file linalg/svd/backends/seq/svd.cpp
/// @brief Sequential one-sided Jacobi SVD.
///
/// See svd/svd.cpp for the full algorithm commentary.
#include "impl.hpp"
#include <cmath>
#include <algorithm>

namespace num::backends::seq {

SVDResult svd(const Matrix& A_in, real tol, idx max_sweeps) {
    constexpr real tiny = 1e-300;
    idx            m = A_in.rows(), n = A_in.cols();
    idx            r = std::min(m, n);

    Matrix A = A_in;

    Matrix V(n, n, 0.0);
    for (idx i = 0; i < n; ++i)
        V(i, i) = 1.0;

    idx  sweeps    = 0;
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
                max_cos     = std::max(max_cos, cos_pq);

                if (cos_pq < tol)
                    continue;

                real zeta = (beta - alpha) / (2.0 * gamma);
                real t    = std::copysign(1.0, zeta)
                         / (std::abs(zeta) + std::sqrt(1.0 + zeta * zeta));
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

} // namespace num::backends::seq
