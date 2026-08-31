/// @file expv.hpp
/// @brief Krylov subspace matrix exponential-vector product: compute exp(t*A)*v
///
/// Approximates \f$\exp(tA)v \approx \|v\| Q_m \exp(tH_m)e_1\f$ where
/// \f$AQ_m \approx Q_{m+1}\bar{H}_m\f$ is the Arnoldi relation.
/// @todo Add adaptive step subdivision and an a posteriori error estimate for
/// large \f$|t|\|A\|\f$.
#pragma once

#include "container/matrix_ops.hpp"
#include "linear/factorization/lu.hpp"
#include "linear/sparse/sparse_op.hpp"
#include <algorithm>
#include <cmath>

#include "container/vector_ops.hpp"

#include "container/matrix.hpp"
#include "core/types.hpp"
#include "container/vector.hpp"
#include "linear/subspace.hpp"
#include "linear/sparse/sparse.hpp"
#include "operator/concepts.hpp"
#include <stdexcept>
#include <utility>
#include <vector>

namespace num {

namespace detail {
Matrix dense_expm_pade6(const Matrix &A);
}

/// @brief Compute \f$\exp(tA)v\f$ for any \f$y=A x\f$ adapter.
template <class Op>
requires LinearOperator<Op, Vector, Vector> Vector expv(real t, const Op &A, const Vector &v,
                                                        int m_max = 30, real tol = 1e-8) {
    const idx n = A.rows();
    if (A.cols() != n || v.size() != n) {
        throw std::invalid_argument("expv: dimension mismatch");
    }

    real beta = norm(v);
    if (beta < 1e-300) {
        return Vector(n, 0.0);
    }

    std::vector<Vector> V;
    V.reserve(m_max + 1);

    Vector v0(n);
    for (idx i = 0; i < n; i++) {
        v0[i] = v[i] / beta;
    }
    V.push_back(std::move(v0));

    Matrix H(m_max + 1, m_max, 0.0);
    int m_actual = m_max;
    std::vector<real> h_col(m_max + 1, 0.0);

    for (int j = 0; j < m_max; j++) {
        Vector w(n, 0.0);
        A.apply(V[j], w);

        const real h_next = kernel::subspace::mgs_orthogonalize(V, w, h_col, j + 1);
        for (int i = 0; i <= j; i++) {
            H(i, j) = h_col[i];
        }
        H(j + 1, j) = h_next;

        if (h_next < tol) {
            m_actual = j + 1;
            break;
        }

        scale(w, real(1) / h_next);
        V.push_back(std::move(w));
    }

    Matrix projected(m_actual, m_actual, 0.0);
    for (int i = 0; i < m_actual; i++) {
        for (int j = 0; j < m_actual; j++) {
            projected(i, j) = t * H(i, j);
        }
    }

    Matrix E = detail::dense_expm_pade6(projected);

    Vector result(n, 0.0);
    for (int j = 0; j < m_actual; j++) {
        axpy(beta * E(j, 0), V[j], result);
    }

    return result;
}

Vector expv(real t, const SparseMatrix &A, const Vector &v, int m_max = 30, real tol = 1e-8);

namespace detail {

inline Matrix dense_expm_pade6(const Matrix &A) {
    const idx m = A.rows();

    static constexpr double c[7] = {1.0,         0.5,           5.0 / 44.0,    1.0 / 66.0,
                                    1.0 / 792.0, 1.0 / 15840.0, 1.0 / 665280.0};

    double norm_inf = 0.0;
    for (idx i = 0; i < m; i++) {
        double row_sum = 0.0;
        for (idx j = 0; j < m; j++) {
            row_sum += std::abs(A(i, j));
        }
        norm_inf = std::max(norm_inf, row_sum);
    }

    int s = 0;
    if (norm_inf > 0.5) {
        s = (int)std::max(0.0, std::ceil(std::log2(norm_inf / 0.5)));
    }

    double scale = std::ldexp(1.0, -s);
    Matrix as(m, m, 0.0);
    // A_s <- 2^{-s} A.
    kernel::raw::scale_copy_strided(as.data(), 1, A.data(), 1, scale, m * m);

    Matrix B(m, m, 0.0);
    matmul(as, as, B);

    Matrix B2(m, m, 0.0);
    matmul(B, B, B2);

    Matrix B3(m, m, 0.0);
    matmul(B2, B, B3);

    Matrix v_mat(m, m, 0.0);
    for (idx i = 0; i < m; i++) {
        for (idx j = 0; j < m; j++) {
            v_mat(i, j) = (c[6] * B3(i, j)) + (c[4] * B2(i, j)) + (c[2] * B(i, j));
        }
        v_mat(i, i) += c[0];
    }

    Matrix W(m, m, 0.0);
    for (idx i = 0; i < m; i++) {
        for (idx j = 0; j < m; j++) {
            W(i, j) = (c[5] * B2(i, j)) + (c[3] * B(i, j));
        }
        W(i, i) += c[1];
    }

    Matrix U(m, m, 0.0);
    matmul(as, W, U);

    Matrix vp_u(m, m, 0.0);
    Matrix vm_u(m, m, 0.0);
    // V_plus <- V + U, V_minus <- V - U.
    kernel::raw::axpbyz(vp_u.data(), v_mat.data(), U.data(), real(1), real(1), m * m);
    kernel::raw::axpbyz(vm_u.data(), v_mat.data(), U.data(), real(1), real(-1), m * m);

    // vm_u is constructed m-by-m.
    LUResult fac = lu(assume_square(vm_u));
    Matrix E(m, m, 0.0);
    lu_solve(fac, vp_u, E);

    for (int i = 0; i < s; i++) {
        Matrix E2(m, m, 0.0);
        matmul(E, E, E2);
        E = std::move(E2);
    }

    return E;
}

} // namespace detail

inline Vector expv(real t, const SparseMatrix &A, const Vector &v, int m_max, real tol) {
    operators::SparseOp op(A);
    return expv(t, op, v, m_max, tol);
}

} // namespace num
