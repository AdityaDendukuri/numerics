/// @file linear/factorization/hessenberg.hpp
/// @brief Upper Hessenberg decomposition A = Q H Q^T via Householder reflections.
#pragma once

#include "container/matrix.hpp"
#include "container/vector.hpp"
#include "core/debug.hpp"
#include "core/policy.hpp"
#include "core/types.hpp"
#include "kernel/complex.hpp"
#include "kernel/kernel.hpp"
#include "lapack/lapack_wrapper.hpp"
#include "linear/factorization/qr.hpp"
#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

namespace num {

// Shifted Hessenberg resolvent

/// @brief Factor \f$sI - H\f$ in place for an upper Hessenberg \f$H\f$, in \f$O(n^2)\f$.
///
/// Only one subdiagonal entry per column is eliminated, and pivoting compares it against the
/// diagonal. This makes each shift of a Krylov resolvent cheap.
///
/// @param work  In/out, n*n. Receives \f$sI - H\f$ and its factors.
/// @param H     Upper Hessenberg matrix, n*n row-major, real.
/// @param shift Complex shift \f$s\f$.
/// @param n     Dimension.
/// @param piv   Output pivot record, length n.
template <std::floating_point T, class Index>
inline void hessenberg_shifted_factor(std::complex<T> *NUM_K_RESTRICT work,
                                      const T *NUM_K_RESTRICT H, std::complex<T> shift, idx n,
                                      Index *NUM_K_RESTRICT piv) noexcept {
    using C = std::complex<T>;
    const T tiny = T(1e-30);

    for (idx i = 0; i < n; ++i) {
        const T *h_row = H + (i * n);
        C *m_row = work + (i * n);
        for (idx j = 0; j < n; ++j) {
            m_row[j] = (i == j ? shift : C(0, 0)) - h_row[j];
        }
    }

    for (idx i = 0; i + 1 < n; ++i) {
        C *row_i = work + (i * n);
        C *row_next = work + ((i + 1) * n);

        if (std::abs(row_next[i]) > std::abs(row_i[i])) {
            for (idx j = i; j < n; ++j) {
                std::swap(row_i[j], row_next[j]);
            }
            piv[i] = static_cast<Index>(i + 1);
        } else {
            piv[i] = static_cast<Index>(i);
        }

        const C pivot = row_i[i];
        if (std::abs(pivot) > tiny) {
            const C mult = row_next[i] / pivot;
            row_next[i] = mult;
            for (idx j = i + 1; j < n; ++j) {
                row_next[j] -= mult * row_i[j];
            }
        }
    }
}

/// @brief Substitute a right-hand side through a factored shifted Hessenberg system.
///
/// Separate from the factorization so that many right-hand sides share one
/// factorization at the same shift. `y` and `b` may alias.
template <std::floating_point T, class Index>
inline void hessenberg_shifted_substitute(std::complex<T> *y,
                                          const std::complex<T> *NUM_K_RESTRICT work,
                                          const Index *NUM_K_RESTRICT piv, const std::complex<T> *b,
                                          idx n) noexcept {
    using C = std::complex<T>;
    const T tiny = T(1e-30);

    for (idx i = 0; i < n; ++i) {
        y[i] = b[i];
    }
    for (idx i = 0; i + 1 < n; ++i) {
        if (static_cast<idx>(piv[i]) != i) {
            std::swap(y[i], y[i + 1]);
        }
        y[i + 1] -= work[((i + 1) * n) + i] * y[i];
    }
    for (idx step = 0; step < n; ++step) {
        const idx i = n - 1 - step;
        const C *row_i = work + (i * n);
        C sum = y[i];
        for (idx j = i + 1; j < n; ++j) {
            sum -= row_i[j] * y[j];
        }
        const C diag = row_i[i];
        y[i] = std::abs(diag) < tiny ? C(0, 0) : sum / diag;
    }
}

/// @brief Solve \f$(sI - H)\,y = b\f$ for a single right-hand side.
template <std::floating_point T, class Index>
inline void hessenberg_shifted_solve(std::complex<T> *y, const T *H, std::complex<T> shift,
                                     const std::complex<T> *b, idx n,
                                     std::complex<T> *NUM_K_RESTRICT work,
                                     Index *NUM_K_RESTRICT piv) noexcept {
    hessenberg_shifted_factor(work, H, shift, n, piv);
    hessenberg_shifted_substitute(y, work, piv, b, n);
}

} // namespace num

namespace num {

/// @brief Upper Hessenberg decomposition of a square matrix: A = Q H Q^T.
class hessenberg_decomposition {
  public:
    /// Compute the Hessenberg decomposition of square matrix A, preferring
    /// LAPACK (`dgehrd`/`dorghr`) when configured, else the in-tree vectorized
    /// Householder elimination. To force one explicitly, call
    /// `num::lapack::hessenberg`/`num::seq::hessenberg`.
    explicit hessenberg_decomposition(const mat<real> &A) : hessenberg_decomposition(A, has_lapack) {}

    /// Selects LAPACK vs. the sequential path explicitly. Prefer the free
    /// functions `num::hessenberg`/`num::lapack::hessenberg`/`num::seq::hessenberg`.
    hessenberg_decomposition(const mat<real> &A, bool use_lapack);

    [[nodiscard]] idx size() const noexcept { return H_.rows(); }
    [[nodiscard]] const mat<real> &H() const noexcept { return H_; }
    [[nodiscard]] const mat<real> &Q() const noexcept { return Q_; }

  private:
    mat<real> H_;
    mat<real> Q_;
};

/// Compute the upper Hessenberg decomposition of a square matrix.
[[nodiscard]] inline hessenberg_decomposition hessenberg(const mat<real> &A) {
    return hessenberg_decomposition(A);
}

namespace lapack {
[[nodiscard]] inline hessenberg_decomposition hessenberg(const mat<real> &A) {
    return hessenberg_decomposition(A, true);
}
} // namespace lapack

namespace seq {
[[nodiscard]] inline hessenberg_decomposition hessenberg(const mat<real> &A) {
    return hessenberg_decomposition(A, false);
}
} // namespace seq

inline hessenberg_decomposition::hessenberg_decomposition(const mat<real> &A, bool use_lapack)
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
        num::householder_vector(v.data(), beta, col_k.data(), m);
        if (beta == 0.0) {
            continue;
        }

        // 1. Left multiplication: H(k+1:n, k:n) <- (I - beta * v * v^T) * H(k+1:n, k:n)
        num::householder_left(&H_raw[((k + 1) * n) + k], n, v.data(), beta, m, n - k, w.data());

        // 2. Right multiplication: H(0:n, k+1:n) <- H(0:n, k+1:n) * (I - beta * v * v^T)
        num::householder_right(&H_raw[k + 1], n, v.data(), beta, n, m);

        // 3. Accumulate into Q: Q(0:n, k+1:n) <- Q(0:n, k+1:n) * (I - beta * v * v^T)
        num::householder_right(&Q_raw[k + 1], n, v.data(), beta, n, m);

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
inline void hessenberg_shifted_solve(const mat<real> &H, cplx shift, const vec<cplx> &b_tilde,
                                     vec<cplx> &y, array<cplx> &M_buf, array<idx> &pivots) {
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
        y = vec<cplx>(n);
    }
    num::hessenberg_shifted_solve(y.data(), H.data(), shift, b_tilde.data(), n, M_buf.data(),
                                  pivots.data());
}

/// @brief Project a right-hand side onto the Hessenberg basis: \f$\tilde b = Q^T b\f$.
///
/// Accepts a real or complex right-hand side. The result is complex either way,
/// since the shift generally is.
template <class Rhs>
inline vec<cplx> hessenberg_project(const mat<real> &Q, const Rhs &b) {
    const idx n = Q.rows();
    vec<cplx> b_tilde(n);
    kernel::matvec_transpose_into_complex(b_tilde.data(), Q.data(), b.data(), n, n);
    return b_tilde;
}

/// @brief Carry a solution back to the original basis: \f$x = Q y\f$.
inline void hessenberg_back_project(const mat<real> &Q, const vec<cplx> &y, vec<cplx> &x) {
    const idx n = Q.rows();
    if (x.size() != n) {
        x = vec<cplx>(n);
    }
    kernel::matvec_real_complex(x.data(), Q.data(), y.data(), n, Q.cols());
}

} // namespace num
