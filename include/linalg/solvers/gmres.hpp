/// @file gmres.hpp
/// @brief Restarted GMRES for general linear systems.
///
/// Minimizes \f$\|b-Ax_k\|_2\f$ over \f$x_0+\mathcal{K}_m(A,r_0)\f$ and
/// restarts after \f$m=\texttt{restart}\f$ Arnoldi steps.
/// @todo Add left/right preconditioned GMRES and flexible GMRES variants.
#pragma once
#include "core/matrix.hpp"
#include "core/policy.hpp"
#include "core/vector.hpp"
#include "kernel/subspace.hpp"
#include "linalg/solvers/solver_result.hpp"
#include "linalg/sparse/sparse.hpp"
#include "operator/concepts.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace num {

/// @brief Operator GMRES for any \f$y=A x\f$ adapter.
template <class Op>
requires LinearOperator<Op, Vector, Vector>
    SolverResult gmres(const Op &A, const Vector &b, Vector &x, real tol = 1e-6,
                       idx max_iter = 1000, idx restart = 30) {
    const idx n = b.size();
    if (A.rows() != n || A.cols() != n || x.size() != n) {
        throw std::invalid_argument("Dimension mismatch in operator GMRES solver");
    }
    if (restart <= 0) {
        throw std::invalid_argument("GMRES restart must be positive");
    }

    restart = std::min(restart, n);
    SolverResult result{0, 0.0, false};

    std::vector<Vector> V;
    V.reserve(restart + 1);
    std::vector<std::vector<real>> H(restart, std::vector<real>(restart + 1, 0.0));
    std::vector<real> cs(restart, 0.0);
    std::vector<real> sn(restart, 0.0);
    std::vector<real> g(restart + 1, 0.0);
    Vector scratch(n);

    idx total_iters = 0;

    while (total_iters < max_iter) {
        Vector r(n);
        A.apply(x, r);
        for (idx i = 0; i < n; ++i) {
            r[i] = b[i] - r[i];
        }

        const real beta = norm(r);
        result.residual = beta;
        if (beta < tol) {
            result.converged = true;
            break;
        }

        V.clear();
        V.emplace_back(n);
        for (idx i = 0; i < n; ++i) {
            V[0][i] = r[i] / beta;
        }

        for (auto &col : H) {
            std::fill(col.begin(), col.end(), 0.0);
        }
        std::fill(cs.begin(), cs.end(), 0.0);
        std::fill(sn.begin(), sn.end(), 0.0);
        std::fill(g.begin(), g.end(), 0.0);
        g[0] = beta;

        idx j = 0;
        for (; j < restart && total_iters < max_iter; ++j, ++total_iters) {
            result.iterations = total_iters + 1;

            A.apply(V[j], scratch);
            const real h_next = kernel::subspace::mgs_orthogonalize(V, scratch, H[j], j + 1);
            H[j][j + 1] = h_next;

            if (h_next > real(1e-15)) {
                scale(scratch, real(1) / h_next);
                V.push_back(scratch);
            } else {
                ++j;
                break;
            }

            for (idx i = 0; i < j; ++i) {
                real tmp = (cs[i] * H[j][i]) + (sn[i] * H[j][i + 1]);
                H[j][i + 1] = (-sn[i] * H[j][i]) + (cs[i] * H[j][i + 1]);
                H[j][i] = tmp;
            }

            real h0 = H[j][j], h1 = H[j][j + 1];
            real denom = std::sqrt((h0 * h0) + (h1 * h1));
            if (denom < real(1e-15)) {
                cs[j] = 1.0;
                sn[j] = 0.0;
            } else {
                cs[j] = h0 / denom;
                sn[j] = h1 / denom;
            }

            H[j][j] = (cs[j] * h0) + (sn[j] * h1);
            H[j][j + 1] = 0.0;

            g[j + 1] = -sn[j] * g[j];
            g[j] = cs[j] * g[j];

            result.residual = std::abs(g[j + 1]);
            if (result.residual < tol) {
                result.converged = true;
                ++j;
                break;
            }
        }

        const idx m = j;
        std::vector<real> y(m, 0.0);
        for (idx i = m; i > 0;) {
            --i;
            y[i] = g[i];
            for (idx k = i + 1; k < m; ++k) {
                y[i] -= H[k][i] * y[k];
            }
            y[i] /= H[i][i];
        }

        for (idx i = 0; i < m; ++i) {
            for (idx k = 0; k < n; ++k) {
                x[k] += y[i] * V[i][k];
            }
        }

        if (result.converged) {
            break;
        }
    }

    return result;
}

/// Solve a stored sparse general system with restarted GMRES.
SolverResult gmres(const SparseMatrix &A, const Vector &b, Vector &x, real tol = 1e-6,
                   idx max_iter = 1000, idx restart = 30);

/// Solve a stored dense general system with restarted GMRES.
SolverResult gmres(const Matrix &A, const Vector &b, Vector &x, real tol = 1e-6,
                   idx max_iter = 1000, idx restart = 30, Backend backend = default_backend);

} // namespace num
