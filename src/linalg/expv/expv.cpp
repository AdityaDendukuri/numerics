/// @file expv.cpp
/// @brief Implementation of Krylov-Pade matrix exponential-vector product
#include "linalg/expv/expv.hpp"
#include "core/matrix.hpp"
#include "core/vector.hpp"
#include "linalg/factorization/lu.hpp"
#include "linalg/sparse/sparse_op.hpp"
#include <algorithm>
#include <cmath>

namespace num {
namespace detail {

Matrix dense_expm_pade6(const Matrix& A) {
    const idx m = A.rows();

    static constexpr double c[7] =
        {1.0, 0.5, 5.0 / 44.0, 1.0 / 66.0, 1.0 / 792.0, 1.0 / 15840.0, 1.0 / 665280.0};

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
    Matrix As(m, m, 0.0);
    for (idx i = 0; i < m; i++) {
        for (idx j = 0; j < m; j++) {
            As(i, j) = A(i, j) * scale;
        }
    }

    Matrix B(m, m, 0.0);
    matmul(As, As, B);

    Matrix B2(m, m, 0.0);
    matmul(B, B, B2);

    Matrix B3(m, m, 0.0);
    matmul(B2, B, B3);

    Matrix V_mat(m, m, 0.0);
    for (idx i = 0; i < m; i++) {
        for (idx j = 0; j < m; j++) {
            V_mat(i, j) = (c[6] * B3(i, j)) + (c[4] * B2(i, j)) + (c[2] * B(i, j));
        }
        V_mat(i, i) += c[0];
    }

    Matrix W(m, m, 0.0);
    for (idx i = 0; i < m; i++) {
        for (idx j = 0; j < m; j++) {
            W(i, j) = (c[5] * B2(i, j)) + (c[3] * B(i, j));
        }
        W(i, i) += c[1];
    }

    Matrix U(m, m, 0.0);
    matmul(As, W, U);

    Matrix VpU(m, m, 0.0);
    Matrix VmU(m, m, 0.0);
    for (idx i = 0; i < m; i++) {
        for (idx j = 0; j < m; j++) {
            VpU(i, j) = V_mat(i, j) + U(i, j);
            VmU(i, j) = V_mat(i, j) - U(i, j);
        }
    }

    LUResult fac = lu(VmU);
    Matrix E(m, m, 0.0);
    lu_solve(fac, VpU, E);

    for (int i = 0; i < s; i++) {
        Matrix E2(m, m, 0.0);
        matmul(E, E, E2);
        matmul(E, E, E2);
        E = std::move(E2);
    }

    return E;
}

} // namespace detail

Vector expv(real t, const SparseMatrix& A, const Vector& v, int m_max, real tol) {
    operators::SparseOp op(A);
    return expv(t, op, v, m_max, tol);
}

} // namespace num
