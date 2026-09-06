/// @file linear/factorization/hessenberg.hpp
/// @brief Upper Hessenberg decomposition A = Q H Q^T via Householder reflections.
#pragma once

#include "container/matrix.hpp"
#include "lapack/lapack_wrapper.hpp"
#include "container/vector.hpp"
#include "core/debug.hpp"
#include "core/policy.hpp"
#include "core/types.hpp"
#include "kernel/complex.hpp"
#include "kernel/factor.hpp"
#include "kernel/kernel.hpp"
#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

namespace num {

/// @brief Upper Hessenberg decomposition of a square matrix: A = Q H Q^T.
class hessenberg_decomposition {
  public:
    /// Compute the Hessenberg decomposition of square matrix A, preferring
    /// LAPACK (`dgehrd`/`dorghr`) when configured, else the in-tree vectorized
    /// Householder elimination. To force one explicitly, call
    /// `num::lapack::hessenberg`/`num::seq::hessenberg`.
    explicit hessenberg_decomposition(const mat &A) : hessenberg_decomposition(A, has_lapack) {}

    /// Selects LAPACK vs. the sequential path explicitly. Prefer the free
    /// functions `num::hessenberg`/`num::lapack::hessenberg`/`num::seq::hessenberg`.
    hessenberg_decomposition(const mat &A, bool use_lapack);

    [[nodiscard]] idx size() const noexcept { return H_.rows(); }
    [[nodiscard]] const mat &H() const noexcept { return H_; }
    [[nodiscard]] const mat &Q() const noexcept { return Q_; }

  private:
    mat H_;
    mat Q_;
};

/// Compute the upper Hessenberg decomposition of a square matrix.
[[nodiscard]] inline hessenberg_decomposition hessenberg(const mat &A) {
    return hessenberg_decomposition(A);
}

namespace lapack {
[[nodiscard]] inline hessenberg_decomposition hessenberg(const mat &A) {
    return hessenberg_decomposition(A, true);
}
} // namespace lapack

namespace seq {
[[nodiscard]] inline hessenberg_decomposition hessenberg(const mat &A) {
    return hessenberg_decomposition(A, false);
}
} // namespace seq

inline hessenberg_decomposition::hessenberg_decomposition(const mat &A, bool use_lapack)
    : H_(A), Q_(A.rows(), A.cols(), 0.0) {
    debug::check_dim(A.rows(), A.cols(), "hessenberg_decomposition matrix must be square");
    debug::check_non_empty(A.rows(), "hessenberg_decomposition matrix");

    const idx n = A.rows();
    // Initialize Q as the identity matrix
    for (idx i = 0; i < n; ++i) {
        Q_(i, i) = 1.0;
    }

    if (n <= 2) {
        return;
    }

#if defined(NUMERICS_HAS_LAPACK)
    if (use_lapack) {
        // LAPACK dgehrd and dorghr assume column-major layout.
        // A (row-major) corresponds to A^T (column-major).
        // Transpose A to column-major buffer:
        array<double> a_col(n * n);
        kernel::transpose(a_col.data(), A.data(), n, n);

        array<double> tau(n - 1, 0.0);
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
        kernel::transpose(Q_.data(), a_col.data(), n, n);
        return;
    }
#endif

    // High-performance vectorized sequential Householder elimination
    array<double> col_k(n);
    array<double> v(n);
    array<double> w(n);
    double *H_raw = H_.data();
    double *Q_raw = Q_.data();

    // Eliminate below subdiagonal column by column
    for (idx k = 0; k < n - 2; ++k) {
        const idx m = n - 1 - k; // length of subvector to reflect

        for (idx i = 0; i < m; ++i) {
            col_k[i] = H_raw[((k + 1 + i) * n) + k];
        }

        double beta = 0.0;
        kernel::householder_vector(v.data(), beta, col_k.data(), m);
        if (beta == 0.0) {
            continue;
        }

        // 1. Left multiplication: H(k+1:n, k:n) <- (I - beta * v * v^T) * H(k+1:n, k:n)
        kernel::householder_left(&H_raw[((k + 1) * n) + k], n, v.data(), beta, m, n - k,
                                      w.data());

        // 2. Right multiplication: H(0:n, k+1:n) <- H(0:n, k+1:n) * (I - beta * v * v^T)
        kernel::householder_right(&H_raw[k + 1], n, v.data(), beta, n, m);

        // 3. Accumulate into Q: Q(0:n, k+1:n) <- Q(0:n, k+1:n) * (I - beta * v * v^T)
        kernel::householder_right(&Q_raw[k + 1], n, v.data(), beta, n, m);

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
inline void hessenberg_shifted_solve(const mat &H, cplx shift, const array<cplx> &b_tilde,
                                     array<cplx> &y, array<cplx> &M_buf,
                                     array<idx> &pivots) {
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
    kernel::hessenberg_shifted_solve(y.data(), H.data(), shift, b_tilde.data(), n,
                                          M_buf.data(), pivots.data());
}

/// @brief Project a right-hand side onto the Hessenberg basis: \f$\tilde b = Q^T b\f$.
///
/// Accepts a real or complex right-hand side. The result is complex either way,
/// since the shift generally is.
template <class Rhs>
inline array<cplx> hessenberg_project(const mat &Q, const Rhs &b) {
    const idx n = Q.rows();
    array<cplx> b_tilde(n);
    kernel::matvec_transpose_into_complex(b_tilde.data(), Q.data(), b.data(), n, n);
    return b_tilde;
}

/// @brief Carry a solution back to the original basis: \f$x = Q y\f$.
inline void hessenberg_back_project(const mat &Q, const array<cplx> &y,
                                    array<cplx> &x) {
    const idx n = Q.rows();
    if (x.size() != n) {
        x.resize(n);
    }
    kernel::matvec_real_complex(x.data(), Q.data(), y.data(), n, Q.cols());
}

} // namespace num
