/// @file linear/factorization/hessenberg.hpp
/// @brief Upper Hessenberg decomposition A = Q H Q^T via Householder reflections.
#pragma once

#include "container/matrix.hpp"
#include "container/parallel/lapack_wrapper.hpp"
#include "container/vector.hpp"
#include "core/debug.hpp"
#include "core/policy.hpp"
#include "core/types.hpp"
#include "kernel/factor.hpp"
#include "kernel/raw.hpp"
#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

namespace num {

/// @brief Upper Hessenberg decomposition of a square matrix: A = Q H Q^T.
class HessenbergDecomposition {
  public:
    /// Compute the Hessenberg decomposition of square matrix A.
    explicit HessenbergDecomposition(const Matrix &A, Backend backend = backend::dflt);

    [[nodiscard]] idx size() const noexcept { return H_.rows(); }
    [[nodiscard]] const Matrix &H() const noexcept { return H_; }
    [[nodiscard]] const Matrix &Q() const noexcept { return Q_; }

  private:
    Matrix H_;
    Matrix Q_;
};

/// Compute the upper Hessenberg decomposition of a square matrix.
[[nodiscard]] inline HessenbergDecomposition hessenberg(const Matrix &A,
                                                        Backend backend = backend::dflt) {
    return HessenbergDecomposition(A, backend);
}



inline HessenbergDecomposition::HessenbergDecomposition(const Matrix &A, Backend backend)
    : H_(A), Q_(A.rows(), A.cols(), 0.0) {
    debug::check_dim(A.rows(), A.cols(), "HessenbergDecomposition matrix must be square");
    debug::check_non_empty(A.rows(), "HessenbergDecomposition matrix");

    const idx n = A.rows();
    // Initialize Q as the identity matrix
    for (idx i = 0; i < n; ++i) {
        Q_(i, i) = 1.0;
    }

    if (n <= 2) {
        return;
    }

#if defined(NUMERICS_HAS_LAPACK)
    if (backend == backend::lapack) {
        // LAPACK dgehrd and dorghr assume column-major layout.
        // A (row-major) corresponds to A^T (column-major).
        // Transpose A to column-major buffer:
        std::vector<double> a_col(n * n);
        kernel::raw::transpose(a_col.data(), A.data(), n, n);

        std::vector<double> tau(n - 1, 0.0);
        lapack_int lapack_n = static_cast<lapack_int>(n);
        lapack_int ilo = 1;
        lapack_int ihi = lapack_n;

        int info = LAPACKE_dgehrd(LAPACK_COL_MAJOR, lapack_n, ilo, ihi, a_col.data(), lapack_n,
                                  tau.data());
        if (info != 0) {
            throw std::runtime_error("dgehrd failed with info=" + std::to_string(info));
        }

        // Copy out upper Hessenberg matrix H (from column-major a_col to row-major H_)
        for (idx i = 0; i < n; ++i) {
            for (idx j = 0; j < n; ++j) {
                if (i > j + 1) {
                    H_(i, j) = 0.0;
                } else {
                    H_(i, j) = a_col[(j * n) + i];
                }
            }
        }

        // Generate orthogonal matrix Q via dorghr
        info = LAPACKE_dorghr(LAPACK_COL_MAJOR, lapack_n, ilo, ihi, a_col.data(), lapack_n,
                              tau.data());
        if (info != 0) {
            throw std::runtime_error("dorghr failed with info=" + std::to_string(info));
        }

        // Copy Q from column-major a_col to row-major Q_
        kernel::raw::transpose(Q_.data(), a_col.data(), n, n);
        return;
    }
#endif

    // High-performance vectorized sequential Householder elimination
    std::vector<double> col_k(n);
    std::vector<double> v(n);
    std::vector<double> w(n);
    double *H_raw = H_.data();
    double *Q_raw = Q_.data();

    // Eliminate below subdiagonal column by column
    for (idx k = 0; k < n - 2; ++k) {
        const idx m = n - 1 - k; // length of subvector to reflect

        for (idx i = 0; i < m; ++i) {
            col_k[i] = H_raw[((k + 1 + i) * n) + k];
        }

        double beta = 0.0;
        kernel::raw::householder_vector(v.data(), beta, col_k.data(), m);
        if (beta == 0.0) {
            continue;
        }

        // 1. Left multiplication: H(k+1:n, k:n) <- (I - beta * v * v^T) * H(k+1:n, k:n)
        kernel::raw::householder_left(&H_raw[((k + 1) * n) + k], n, v.data(), beta, m, n - k,
                                      w.data());

        // 2. Right multiplication: H(0:n, k+1:n) <- H(0:n, k+1:n) * (I - beta * v * v^T)
        kernel::raw::householder_right(&H_raw[k + 1], n, v.data(), beta, n, m);

        // 3. Accumulate into Q: Q(0:n, k+1:n) <- Q(0:n, k+1:n) * (I - beta * v * v^T)
        kernel::raw::householder_right(&Q_raw[k + 1], n, v.data(), beta, n, m);

        // Set strictly zero entries below subdiagonal
        for (idx i = k + 2; i < n; ++i) {
            H_raw[(i * n) + k] = 0.0;
        }
    }
}


/// @brief Solve the shifted Hessenberg system \f$(sI - H)y = \tilde b\f$ in \f$O(n^2)\f$.
///
/// Gaussian elimination with partial pivoting on an upper Hessenberg matrix touches
/// one subdiagonal per column, so the factorization is \f$O(n^2)\f$ rather than
/// \f$O(n^3)\f$. This is what makes a resolvent \f$(sI-A)^{-1}b\f$ cheap once
/// \f$A\f$ has been reduced once: every subsequent shift reuses the same \f$H\f$.
///
/// @param H Upper Hessenberg matrix, n*n.
/// @param shift The scalar s.
/// @param b_tilde Right-hand side, length n.
/// @param y Output, resized to n.
/// @param M_buf Scratch, grown to n*n and reusable across calls.
/// @param pivots Scratch, grown to n and reusable across calls.
inline void hessenberg_shifted_solve(const Matrix &H, cplx shift, const std::vector<cplx> &b_tilde,
                                     std::vector<cplx> &y, std::vector<cplx> &M_buf,
                                     std::vector<idx> &pivots) {
    const idx n = H.rows();
    if (b_tilde.size() != n) {
        throw std::invalid_argument("hessenberg_shifted_solve: dimension mismatch");
    }
    if (M_buf.size() < n * n) {
        M_buf.resize(n * n);
    }
    if (pivots.size() < n) {
        pivots.resize(n);
    }
    if (y.size() != n) {
        y.resize(n);
    }
    kernel::raw::hessenberg_shifted_solve(y.data(), H.data(), shift, b_tilde.data(), n,
                                          M_buf.data(), pivots.data());
}

/// @brief Project a right-hand side onto the Hessenberg basis: \f$\tilde b = Q^T b\f$.
///
/// Accepts a real or complex right-hand side. The result is complex either way,
/// since the shift generally is.
template <class Rhs>
inline std::vector<cplx> hessenberg_project(const Matrix &Q, const Rhs &b) {
    const idx n = Q.rows();
    std::vector<cplx> b_tilde(n);
    kernel::raw::matvec_transpose_into_complex(b_tilde.data(), Q.data(), b.data(), n, n);
    return b_tilde;
}

/// @brief Carry a solution back to the original basis: \f$x = Q y\f$.
inline void hessenberg_back_project(const Matrix &Q, const std::vector<cplx> &y,
                                    std::vector<cplx> &x) {
    const idx n = Q.rows();
    if (x.size() != n) {
        x.resize(n);
    }
    kernel::raw::matvec_real_complex(x.data(), Q.data(), y.data(), n, Q.cols());
}

} // namespace num
