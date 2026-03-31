/// @file expv.cpp
/// @brief Implementation of Krylov-Pade matrix exponential-vector product
#include "linalg/expv/expv.hpp"
#include "core/matrix.hpp"
#include "linalg/factorization/lu.hpp"
#include "core/vector.hpp"
#include <cmath>
#include <algorithm>
#include <vector>

namespace num {

/// @brief Dense matrix exponential via scaling+squaring + Pade [6/6]
///
/// Pade [6/6] numerator/denominator coefficients:
///   c[0]=1, c[1]=1/2, c[2]=5/44, c[3]=1/66,
///   c[4]=1/792, c[5]=1/15840, c[6]=1/665280
static Matrix dense_expm(const Matrix& A) {
    const idx m = A.rows();

    // Pade [6/6] coefficients
    static constexpr double c[7] = {1.0,
                                    0.5,
                                    5.0 / 44.0,
                                    1.0 / 66.0,
                                    1.0 / 792.0,
                                    1.0 / 15840.0,
                                    1.0 / 665280.0};

    // 1. Compute inf-norm (max absolute row sum)
    double norm_inf = 0.0;
    for (idx i = 0; i < m; i++) {
        double row_sum = 0.0;
        for (idx j = 0; j < m; j++)
            row_sum += std::abs(A(i, j));
        norm_inf = std::max(norm_inf, row_sum);
    }

    // 2. Scaling: s = max(0, ceil(log2(norm_inf / 0.5)))
    int s = 0;
    if (norm_inf > 0.5) {
        s = (int)std::max(0.0, std::ceil(std::log2(norm_inf / 0.5)));
    }

    // 3. As = A / 2^s
    double scale = std::ldexp(1.0, -s); // 1/2^s
    Matrix As(m, m, 0.0);
    for (idx i = 0; i < m; i++)
        for (idx j = 0; j < m; j++)
            As(i, j) = A(i, j) * scale;

    // 4. Build powers: B = As^2, B2 = As^4, B3 = As^6
    Matrix B(m, m, 0.0);
    matmul(As, As, B); // B = As^2

    Matrix B2(m, m, 0.0);
    matmul(B, B, B2); // B2 = As^4

    Matrix B3(m, m, 0.0);
    matmul(B2, B, B3); // B3 = As^6

    // 5. V_mat = c[6]*B3 + c[4]*B2 + c[2]*B + c[0]*I  (even terms)
    Matrix V_mat(m, m, 0.0);
    for (idx i = 0; i < m; i++) {
        for (idx j = 0; j < m; j++) {
            V_mat(i, j) = c[6] * B3(i, j) + c[4] * B2(i, j) + c[2] * B(i, j);
        }
        V_mat(i, i) += c[0];
    }

    // 6. W = c[5]*B2 + c[3]*B + c[1]*I  (inner odd terms, without As factor)
    Matrix W(m, m, 0.0);
    for (idx i = 0; i < m; i++) {
        for (idx j = 0; j < m; j++) {
            W(i, j) = c[5] * B2(i, j) + c[3] * B(i, j);
        }
        W(i, i) += c[1];
    }

    // 7. U = As * W  (full odd numerator terms)
    Matrix U(m, m, 0.0);
    matmul(As, W, U);

    // 8. VpU = V_mat + U,  VmU = V_mat - U
    Matrix VpU(m, m, 0.0);
    Matrix VmU(m, m, 0.0);
    for (idx i = 0; i < m; i++) {
        for (idx j = 0; j < m; j++) {
            VpU(i, j) = V_mat(i, j) + U(i, j);
            VmU(i, j) = V_mat(i, j) - U(i, j);
        }
    }

    // 9. Solve VmU * E = VpU  (Pade approximant: E = (V-U)^{-1}*(V+U))
    LUResult fac = lu(VmU);
    Matrix   E(m, m, 0.0);
    lu_solve(fac, VpU, E);

    // 10. Squaring: E = E^(2^s)
    for (int i = 0; i < s; i++) {
        Matrix E2(m, m, 0.0);
        matmul(E, E, E2);
        E = std::move(E2);
    }

    return E;
}

Vector expv(real            t,
            const MatVecFn& matvec,
            idx             n,
            const Vector&   v,
            int             m_max,
            real            tol) {
    // Handle zero vector
    real beta = norm(v);
    if (beta < 1e-300) {
        return Vector(n, 0.0);
    }

    // Krylov basis vectors: V[0..m_max]
    std::vector<Vector> V;
    V.reserve(m_max + 1);

    // V[0] = v / beta
    Vector v0(n);
    for (idx i = 0; i < n; i++)
        v0[i] = v[i] / beta;
    V.push_back(std::move(v0));

    // Upper Hessenberg matrix H: (m_max+1) x m_max
    Matrix H(m_max + 1, m_max, 0.0);

    int m_actual = m_max;

    // Arnoldi process
    for (int j = 0; j < m_max; j++) {
        // w = A * V[j]
        Vector w(n, 0.0);
        matvec(V[j], w);

        // Modified Gram-Schmidt orthogonalization
        for (int i = 0; i <= j; i++) {
            real h  = dot(V[i], w);
            H(i, j) = h;
            axpy(-h, V[i], w); // w -= h * V[i]
        }

        real h_next = norm(w);
        H(j + 1, j) = h_next;

        // Check for lucky breakdown
        if (h_next < tol) {
            m_actual = j + 1;
            break;
        }

        // Normalize and store next Krylov vector
        Vector vj1(n);
        for (idx i = 0; i < n; i++)
            vj1[i] = w[i] / h_next;
        V.push_back(std::move(vj1));
    }

    // Extract H_m = t * H[0..m_actual-1, 0..m_actual-1]
    Matrix Hm(m_actual, m_actual, 0.0);
    for (int i = 0; i < m_actual; i++)
        for (int j = 0; j < m_actual; j++)
            Hm(i, j) = t * H(i, j);

    // Compute dense expm of Hm
    Matrix E = dense_expm(Hm);

    // result = sum_j (beta * E[j,0]) * V[j]
    Vector result(n, 0.0);
    for (int j = 0; j < m_actual; j++) {
        real coeff = beta * E(j, 0);
        axpy(coeff, V[j], result);
    }

    return result;
}

Vector expv(real                t,
            const SparseMatrix& A,
            const Vector&       v,
            int                 m_max,
            real                tol) {
    MatVecFn fn = [&A](const Vector& x, Vector& y) {
        sparse_matvec(A, x, y);
    };
    return expv(t, fn, A.n_rows(), v, m_max, tol);
}

} // namespace num
