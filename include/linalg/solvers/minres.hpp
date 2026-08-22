/// @file solvers/minres.hpp
/// @brief Minimum residual iteration for symmetric linear systems.
/// @todo Replace the projection-based implementation with the standard
/// short-recurrence MINRES iteration.
#pragma once

#include "core/policy.hpp"
#include "core/vector.hpp"
#include "linalg/factorization/qr.hpp"
#include "linalg/solvers/solver_result.hpp"
#include "operator/concepts.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace num {

namespace detail {

inline Vector minres_projected_solve(const std::vector<real> &alpha, const std::vector<real> &beta,
                                     real beta0, idx m, Backend backend) {
    Matrix H(m + 1, m, 0.0);
    for (idx j = 0; j < m; ++j) {
        H(j, j) = alpha[j];
        if (j > 0) {
            H(j - 1, j) = beta[j - 1];
        }
        H(j + 1, j) = beta[j];
    }

    Vector rhs(m + 1, 0.0);
    rhs[0] = beta0;
    QRResult qrf = qr(H, backend);
    Vector y(m, 0.0);
    qr_solve(qrf, rhs, y);
    return y;
}

} // namespace detail

template <class Op>
requires SymmetricLinearOperator<Op, Vector, Vector>
    /// Solve a symmetric operator system by minimum-residual projection.
    SolverResult minres(const Op &A, const Vector &b, Vector &x, real tol = 1e-10,
                        idx max_iter = 1000, Backend backend = default_backend) {
    const idx n = b.size();
    if (A.rows() != n || A.cols() != n || x.size() != n) {
        throw std::invalid_argument("minres: dimension mismatch");
    }

    Vector r0(n), Ax(n);
    A.apply(x, Ax);
    for (idx i = 0; i < n; ++i) {
        r0[i] = b[i] - Ax[i];
    }

    const real beta0 = norm(r0, backend);
    SolverResult result{0, beta0, beta0 < tol};
    if (result.converged) {
        return result;
    }

    const idx mmax = std::min(max_iter, n);
    std::vector<Vector> V;
    V.reserve(mmax + 1);
    V.emplace_back(n, 0.0);
    for (idx i = 0; i < n; ++i) {
        V[0][i] = r0[i] / beta0;
    }

    std::vector<real> alpha;
    std::vector<real> beta;
    alpha.reserve(mmax);
    beta.reserve(mmax);

    Vector w(n), q_prev(n, 0.0);
    for (idx j = 0; j < mmax; ++j) {
        result.iterations = j + 1;
        A.apply(V[j], w);
        if (j > 0) {
            axpy(-beta[j - 1], q_prev, w, backend);
        }

        const real a = dot(V[j], w, backend);
        alpha.push_back(a);
        axpy(-a, V[j], w, backend);

        const real bnext = norm(w, backend);
        beta.push_back(bnext);

        Vector y = detail::minres_projected_solve(alpha, beta, beta0, j + 1, backend);
        Vector x_candidate = x;
        for (idx col = 0; col <= j; ++col) {
            axpy(y[col], V[col], x_candidate, backend);
        }

        A.apply(x_candidate, Ax);
        real rsq = 0.0;
        for (idx i = 0; i < n; ++i) {
            const real ri = b[i] - Ax[i];
            rsq += ri * ri;
        }
        result.residual = std::sqrt(rsq);
        if (result.residual < tol) {
            x = std::move(x_candidate);
            result.converged = true;
            break;
        }

        if (bnext < real(1e-15)) {
            x = std::move(x_candidate);
            break;
        }

        q_prev = V[j];
        scale(w, real(1) / bnext, backend);
        V.push_back(w);

        if (j + 1 == mmax) {
            x = std::move(x_candidate);
        }
    }

    return result;
}

} // namespace num
