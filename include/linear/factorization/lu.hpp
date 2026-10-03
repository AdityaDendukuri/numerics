/// @file lu.hpp
/// @brief LU factorization, with partial pivoting or without, in any floating-point precision.
#pragma once

#include "container/matrix.hpp"
#include "core/debug.hpp"
#include "core/policy.hpp"
#include "core/types.hpp"
#include "kernel/dense.hpp"
#include "kernel/kernel.hpp"
#include "lapack/lapack_wrapper.hpp"
#include "linear/matrix_properties.hpp"
#include "linear/solvers/solver_result.hpp"
#include "operator/concepts.hpp"
#include "linear/matrix_utils.hpp"
#include <algorithm>
#include <cmath>
#include <concepts>
#include <ostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace num {

/// @brief LU factorization \f$PA = LU\f$ with partial pivoting, packed in place.
///
/// On return `LU` holds the unit-lower factor below the diagonal and the upper
/// factor on and above it; `piv[i]` is the row swapped with row i. Returns false
/// when an exactly zero pivot column is found, i.e. when A is singular.
///
/// @param LU In/out packed factors, n*n row-major; pass a copy of A.
/// @param piv Output pivot sequence, length n.
/// @param n mat dimension.
template <std::floating_point T, class Index>
[[nodiscard]] inline bool lu_factor(T *NUM_K_RESTRICT LU, Index *NUM_K_RESTRICT piv,
                                    idx n) noexcept {
    bool nonsingular = true;
    for (idx k = 0; k < n; ++k) {
        idx pivot_row = k;
        T best = std::abs(LU[(k * n) + k]);
        for (idx i = k + 1; i < n; ++i) {
            const T candidate = std::abs(LU[(i * n) + k]);
            if (candidate > best) {
                best = candidate;
                pivot_row = i;
            }
        }
        piv[k] = static_cast<Index>(pivot_row);

        if (pivot_row != k) {
            kernel::swap_rows(LU, n, k, pivot_row, n);
        }
        const T pivot = LU[(k * n) + k];
        if (pivot == T(0)) {
            nonsingular = false;
            continue;
        }
        for (idx i = k + 1; i < n; ++i) {
            const T factor = LU[(i * n) + k] / pivot;
            LU[(i * n) + k] = factor;
            for (idx j = k + 1; j < n; ++j) {
                LU[(i * n) + j] -= factor * LU[(k * n) + j];
            }
        }
    }
    return nonsingular;
}

/// @brief Blocked variant of `lu_factor`: same packed \f$PA = LU\f$ contract, panel-at-a-time.
template <std::floating_point T, class Index>
[[nodiscard]] inline bool lu_factor_blocked(T *NUM_K_RESTRICT LU, Index *NUM_K_RESTRICT piv, idx n,
                                            idx block_size = 64) noexcept {
    if (block_size == 0)
        block_size = 1;
    bool nonsingular = true;
    for (idx kk = 0; kk < n; kk += block_size) {
        const idx kb = std::min(block_size, n - kk), panel_end = kk + kb;
        for (idx k = kk; k < panel_end; ++k) {
            idx pivot_row = k;
            T best = std::abs(LU[(k * n) + k]);
            for (idx i = k + 1; i < n; ++i) {
                const T candidate = std::abs(LU[(i * n) + k]);
                if (candidate > best) {
                    best = candidate;
                    pivot_row = i;
                }
            }
            piv[k] = static_cast<Index>(pivot_row);
            if (pivot_row != k)
                kernel::swap_rows(LU, n, k, pivot_row, n);
            const T pivot = LU[(k * n) + k];
            if (pivot == T(0)) {
                nonsingular = false;
                continue;
            }
            for (idx i = k + 1; i < n; ++i) {
                const T factor = LU[(i * n) + k] / pivot;
                LU[(i * n) + k] = factor;
                for (idx j = k + 1; j < panel_end; ++j)
                    LU[(i * n) + j] -= factor * LU[(k * n) + j];
            }
        }
        const idx trailing = n - panel_end;
        if (trailing == 0)
            continue;
        // U12 <- L11^{-1} A12.
        kernel::trsm_unit_lower_inplace(LU + (kk * n) + panel_end, n, LU + (kk * n) + kk, n, kb,
                                        trailing);
        // A22 <- A22 - L21*U12.
        kernel::gemm(LU + (panel_end * n) + panel_end, n, LU + (panel_end * n) + kk, n,
                     LU + (kk * n) + panel_end, n, T(-1), T(1), trailing, trailing, kb);
    }
    return nonsingular;
}

/// @brief Solve \f$A x = b\f$ from a packed \f$PA = LU\f$ factorization.
template <std::floating_point T, class Index>
inline void lu_solve(T *NUM_K_RESTRICT x, const T *NUM_K_RESTRICT LU,
                     const Index *NUM_K_RESTRICT piv, const T *b, idx n) noexcept {
    for (idx i = 0; i < n; ++i) {
        x[i] = b[i];
    }
    for (idx k = 0; k < n; ++k) {
        const idx p = static_cast<idx>(piv[k]);
        if (p != k) {
            const T tmp = x[k];
            x[k] = x[p];
            x[p] = tmp;
        }
    }
    // Unit-lower forward substitution, then upper back substitution.
    for (idx i = 1; i < n; ++i) {
        T sum = x[i];
        for (idx j = 0; j < i; ++j) {
            sum -= LU[(i * n) + j] * x[j];
        }
        x[i] = sum;
    }
    for (idx i = n; i-- > 0;) {
        T sum = x[i];
        for (idx j = i + 1; j < n; ++j) {
            sum -= LU[(i * n) + j] * x[j];
        }
        x[i] = sum / LU[(i * n) + i];
    }
}

/// @brief Inverse from packed LU factors: `inverse <- A^{-1}`, one column at a time.
///
/// Column `j` of the inverse is the solve `A x = e_j`, so this is `n` calls
/// of `lu_solve` written without the per-column copy. Out of place: every
/// column's solve reads the whole factor, so the result cannot overwrite it.
/// `work` holds `n` elements.
template <std::floating_point T, class Index>
inline void lu_invert(T *NUM_K_RESTRICT inverse, const T *NUM_K_RESTRICT LU,
                      const Index *NUM_K_RESTRICT piv, idx n, T *NUM_K_RESTRICT work) noexcept {
    for (idx col = 0; col < n; ++col) {
        for (idx i = 0; i < n; ++i)
            work[i] = (i == col) ? T(1) : T(0);
        for (idx k = 0; k < n; ++k) {
            const idx p = static_cast<idx>(piv[k]);
            if (p != k)
                std::swap(work[k], work[p]);
        }
        for (idx i = 1; i < n; ++i) {
            T sum = work[i];
            for (idx j = 0; j < i; ++j)
                sum -= LU[(i * n) + j] * work[j];
            work[i] = sum;
        }
        for (idx i = n; i-- > 0;) {
            T sum = work[i];
            for (idx j = i + 1; j < n; ++j)
                sum -= LU[(i * n) + j] * work[j];
            work[i] = sum / LU[(i * n) + i];
        }
        for (idx i = 0; i < n; ++i) {
            inverse[(i * n) + col] = work[i];
        }
    }
}

/// @brief In-place LU without row pivoting, for matrices such as M-matrices whose pivots
/// stay nonzero. Returns false if a pivot fell below tolerance.
template <std::floating_point T>
[[nodiscard]] inline bool lu_no_pivot(T *A, idx n) noexcept {
    constexpr T tolerance = T(1e-15);
    bool nonsingular = true;
    for (idx k = 0; k < n; ++k) {
        if (std::abs(A[k * n + k]) < tolerance) {
            nonsingular = false;
            continue;
        }
        const T inverse_pivot = T(1) / A[k * n + k];
        for (idx i = k + 1; i < n; ++i) {
            A[i * n + k] *= inverse_pivot;
            const T multiplier = A[i * n + k];
            for (idx j = k + 1; j < n; ++j)
                A[i * n + j] -= multiplier * A[k * n + j];
        }
    }
    return nonsingular;
}

/// @brief Solves several right-hand sides from an unpivoted LU factor.
///
/// One right-hand side is the common case, and there each row of a substitution is a dot
/// product over a contiguous row of the factor: `dot` keeps several partial sums, where an
/// update of `X[i]` in memory at every step would chain each multiply-add to the last.
template <std::floating_point T>
inline void lu_no_pivot_solve_multiple(T *X, const T *LU, idx n, idx columns) noexcept {
    if (columns == 1) {
        for (idx i = 1; i < n; ++i) {
            X[i] -= kernel::dot(LU + (i * n), X, i);
        }
        for (idx i = n; i-- > 0;) {
            const T *row = LU + (i * n);
            X[i] = (X[i] - kernel::dot(row + i + 1, X + i + 1, n - i - 1)) / row[i];
        }
        return;
    }
    for (idx i = 0; i < n; ++i)
        for (idx j = 0; j < i; ++j)
            for (idx c = 0; c < columns; ++c)
                X[i * columns + c] -= LU[i * n + j] * X[j * columns + c];
    for (idx i = n; i-- > 0;) {
        for (idx j = i + 1; j < n; ++j)
            for (idx c = 0; c < columns; ++c)
                X[i * columns + c] -= LU[i * n + j] * X[j * columns + c];
        for (idx c = 0; c < columns; ++c)
            X[i * columns + c] /= LU[i * n + i];
    }
}

/// @brief Solves \f$A^T x = b\f$ for several right-hand sides from an unpivoted LU factor.
///
/// For one right-hand side, \f$U^T\f$ and \f$L^T\f$ are applied a row of the factor at a
/// time: once \f$x_i\f$ is final, row \f$i\f$ of the factor times \f$x_i\f$ is subtracted from
/// the entries still pending. That reads the factor with stride 1 and has no dependence
/// between iterations, where the column-oriented form reads it with stride \f$n\f$.
template <std::floating_point T>
inline void lu_no_pivot_solve_transpose_multiple(T *X, const T *LU, idx n, idx columns) noexcept {
    if (columns == 1) {
        for (idx i = 0; i < n; ++i) {
            const T *NUM_K_RESTRICT row = LU + (i * n);
            X[i] /= row[i];
            const T xi = X[i];
            T *NUM_K_RESTRICT pending = X + i + 1;
            NUM_K_IVDEP
            for (idx j = 0; j < n - i - 1; ++j) {
                pending[j] -= row[i + 1 + j] * xi;
            }
        }
        for (idx i = n; i-- > 1;) {
            const T *NUM_K_RESTRICT row = LU + (i * n);
            const T xi = X[i];
            NUM_K_IVDEP
            for (idx j = 0; j < i; ++j) {
                X[j] -= row[j] * xi;
            }
        }
        return;
    }
    for (idx i = 0; i < n; ++i) {
        for (idx j = 0; j < i; ++j)
            for (idx c = 0; c < columns; ++c)
                X[i * columns + c] -= LU[j * n + i] * X[j * columns + c];
        for (idx c = 0; c < columns; ++c)
            X[i * columns + c] /= LU[i * n + i];
    }
    for (idx i = n; i-- > 0;)
        for (idx j = i + 1; j < n; ++j)
            for (idx c = 0; c < columns; ++c)
                X[i * columns + c] -= LU[j * n + i] * X[j * columns + c];
}

} // namespace num

namespace num {

/// @brief The factorization \f$PA = LU\f$, in precision `T`.
///
/// Both factors share the matrix `LU`, the layout LAPACK `getrf` returns: \f$U\f$ on and
/// above the diagonal, \f$L\f$ strictly below it. \f$L\f$ has a unit diagonal, which is not
/// stored. `lower()` and `upper()` copy the two factors out.
///
/// \f$P\f$ is stored as the row swaps that built it: step `k` exchanged rows `k` and
/// `swaps[k]`. `swaps` is empty when the factorization did not pivot, so \f$P = I\f$.
template <std::floating_point T>
struct lu_result {
    mat<T> LU;
    array<idx> swaps;
    bool singular = false; ///< True when a pivot fell below tolerance.

    [[nodiscard]] idx size() const { return LU.rows(); }

    friend std::ostream &operator<<(std::ostream &os, const lu_result &r) {
        os << "lu_result{ dim: " << r.LU.rows() << "x" << r.LU.cols()
           << ", pivoted: " << (r.swaps.empty() ? "false" : "true")
           << ", singular: " << (r.singular ? "true" : "false") << " }";
        return os;
    }
};

/// @brief The tag type of `num::no_pivot`.
struct no_pivot_structure {};
/// @brief Select LU without pivoting, for matrices whose structure keeps the pivots nonzero.
inline constexpr no_pivot_structure no_pivot{};

namespace seq {
/// Blocked partial-pivoting LU through `num::lu_factor_blocked`: panel
/// factorization, then a `trsm` and a `gemm` per panel, so the bulk of the
/// work runs at `gemm` speed. A pivot below `singular_tol` marks the result
/// singular; the factorization still completes so the caller can inspect it.
template <std::floating_point T>
inline lu_result<T> lu(const mat<T> &A) {
    constexpr T singular_tol = T(1e-14);
    const idx n = A.rows();
    lu_result<T> f;
    f.LU = A;
    f.swaps.resize(n);
    f.singular = !num::lu_factor_blocked(f.LU.data(), f.swaps.data(), n);
    for (idx k = 0; k < n && !f.singular; ++k) {
        f.singular = std::abs(f.LU(k, k)) < singular_tol;
    }
    return f;
}
} // namespace seq

namespace lapack {
inline lu_result<real> lu(const mat<real> &A) {
#if defined(NUMERICS_HAS_LAPACK)
    const idx n = A.rows();
    lu_result<real> f;
    f.LU = A;
    f.swaps.resize(n);
    f.singular = false;

    array<lapack_int> ipiv(n);
    int info =
        LAPACKE_dgetrf(LAPACK_ROW_MAJOR, static_cast<lapack_int>(n), static_cast<lapack_int>(n),
                       f.LU.data(), static_cast<lapack_int>(n), ipiv.data());
    if (info < 0) {
        throw std::runtime_error("lu (lapack): dgetrf argument error, info=" +
                                 std::to_string(info));
    }
    if (info > 0) {
        f.singular = true;
    }

    for (idx k = 0; k < n; ++k) {
        f.swaps[k] = static_cast<idx>(ipiv[k] - 1);
    }

    return f;
#else
    return seq::lu(A);
#endif
}
} // namespace lapack

/// @brief Factor \f$PA = LU\f$ with partial pivoting.
///
/// In double precision, uses LAPACK `dgetrf` above `lapack_factor_threshold` when an optimized
/// LAPACK is configured, else the in-tree blocked kernel; other precisions always use the
/// kernel. Call `num::lapack::lu` or `num::seq::lu` to force one. A singular `A` sets
/// `singular`; the factorization still completes.
/// @throws std::invalid_argument If `A` is not square.
template <std::floating_point T>
inline lu_result<T> lu(const mat<T> &A) {
    if (A.rows() != A.cols()) {
        throw std::invalid_argument("lu: matrix must be square");
    }
#if defined(NUMERICS_LAPACK_DEFAULT)
    if constexpr (std::is_same_v<T, real>) {
        if (A.rows() > lapack_factor_threshold) {
            return lapack::lu(A);
        }
    }
#endif
    return seq::lu(A);
}

/// @brief Factor \f$A = LU\f$ without row swaps, for matrices such as diagonally dominant
/// M-matrices whose pivots stay nonzero.
///
/// Saves the pivot search and row swap at every step. `singular` reports a zero pivot, but
/// small pivots go unchecked, so use `lu(A)` without that structural guarantee.
/// @throws std::invalid_argument If `A` is not square.
template <std::floating_point T>
inline lu_result<T> lu(const mat<T> &A, no_pivot_structure) {
    if (A.rows() != A.cols()) {
        throw std::invalid_argument("lu: matrix must be square");
    }
    lu_result<T> f;
    f.LU = A;
    f.singular = !num::lu_no_pivot(f.LU.data(), f.LU.rows());
    return f;
}

namespace detail {

// X <- P X, applying the swaps in the order the factorization made them.
template <std::floating_point T>
inline void apply_swaps(const lu_result<T> &f, T *X, idx nrhs) {
    for (idx k = 0; k < f.swaps.size(); ++k) {
        if (f.swaps[k] != k) {
            kernel::swap_rows(X, nrhs, k, f.swaps[k], nrhs);
        }
    }
}

// X <- P^T X, undoing the swaps in reverse order.
template <std::floating_point T>
inline void undo_swaps(const lu_result<T> &f, T *X, idx nrhs) {
    for (idx k = f.swaps.size(); k-- > 0;) {
        if (f.swaps[k] != k) {
            kernel::swap_rows(X, nrhs, k, f.swaps[k], nrhs);
        }
    }
}

template <std::floating_point T>
inline void check_rhs(const lu_result<T> &f, idx rows) {
    if (f.LU.cols() != f.size() || rows != f.size()) {
        throw std::invalid_argument("lu solve: dimension mismatch");
    }
}

} // namespace detail

/// @brief Solve \f$Ax = b\f$. `x` may be `b`.
template <std::floating_point T>
inline void solve(const lu_result<T> &f, const std::type_identity_t<vec<T>> &b, vec<T> &x) {
    detail::check_rhs(f, b.size());
    x = b;
    detail::apply_swaps(f, x.data(), 1);
    num::lu_no_pivot_solve_multiple(x.data(), f.LU.data(), f.size(), 1);
}

/// @brief Solve \f$AX = B\f$. `X` may be `B`.
template <std::floating_point T>
inline void solve(const lu_result<T> &f, const std::type_identity_t<mat<T>> &B, mat<T> &X) {
    detail::check_rhs(f, B.rows());
    X = B;
    const idx n = f.size();
    const idx nrhs = B.cols();
    if (f.swaps.empty()) {
        num::lu_no_pivot_solve_multiple(X.data(), f.LU.data(), n, nrhs);
        return;
    }
    // P B, then L Y = P B and U X = Y, both as blocked triangular solves.
    // Never dgetrs: the kernel solve measured faster at every size, and on a
    // threaded BLAS the LAPACK call pays a thread wake-up per solve.
    detail::apply_swaps(f, X.data(), nrhs);
    kernel::trsm_unit_lower_inplace(X.data(), nrhs, f.LU.data(), n, n, nrhs);
    kernel::trsm_upper_inplace(X.data(), nrhs, f.LU.data(), n, n, nrhs);
}

/// @brief Solve \f$A^T x = b\f$. `x` may be `b`.
///
/// \f$A^T = U^T L^T P\f$, so \f$U^T Q = B\f$, \f$L^T Y = Q\f$, \f$X = P^T Y\f$.
template <std::floating_point T>
inline void solve_transpose(const lu_result<T> &f, const std::type_identity_t<vec<T>> &b,
                            vec<T> &x) {
    detail::check_rhs(f, b.size());
    x = b;
    num::lu_no_pivot_solve_transpose_multiple(x.data(), f.LU.data(), f.size(), 1);
    detail::undo_swaps(f, x.data(), 1);
}

/// @brief Solve \f$A^T X = B\f$. `X` may be `B`.
template <std::floating_point T>
inline void solve_transpose(const lu_result<T> &f, const std::type_identity_t<mat<T>> &B,
                            mat<T> &X) {
    detail::check_rhs(f, B.rows());
    X = B;
    const idx n = f.size();
    const idx nrhs = B.cols();
    if (f.swaps.empty()) {
        num::lu_no_pivot_solve_transpose_multiple(X.data(), f.LU.data(), n, nrhs);
        return;
    }
    kernel::trsm_upper_transpose_inplace(X.data(), nrhs, f.LU.data(), n, n, nrhs);
    kernel::trsm_unit_lower_transpose_inplace(X.data(), nrhs, f.LU.data(), n, n, nrhs);
    detail::undo_swaps(f, X.data(), nrhs);
}

/// @brief \f$\det(A) = \det(P)^{-1}\prod_i U_{ii}\f$.
template <std::floating_point T>
inline T det(const lu_result<T> &f) {
    T product = T(1);
    for (idx i = 0; i < f.size(); ++i) {
        product *= f.LU(i, i);
    }
    idx exchanges = 0;
    for (idx k = 0; k < f.swaps.size(); ++k) {
        if (f.swaps[k] != k) {
            ++exchanges;
        }
    }
    return (exchanges % 2 == 0) ? product : -product;
}

/// @brief \f$A^{-1}\f$.
template <std::floating_point T>
inline mat<T> inverse(const lu_result<T> &f) {
    const idx n = f.size();
    array<idx> swaps = f.swaps;
    if (swaps.empty()) {
        swaps.resize(n);
        for (idx k = 0; k < n; ++k) {
            swaps[k] = k;
        }
    }
    mat<T> inv = f.LU;
#if defined(NUMERICS_LAPACK_DEFAULT)
    if constexpr (std::is_same_v<T, real>) {
        array<lapack_int> ipiv(n);
        for (idx i = 0; i < n; ++i)
            ipiv[i] = static_cast<lapack_int>(swaps[i] + 1);
        LAPACKE_dgetri(LAPACK_ROW_MAJOR, static_cast<lapack_int>(n), inv.data(),
                       static_cast<lapack_int>(n), ipiv.data());
        return inv;
    }
#endif
    array<T> work(n);
    num::lu_invert(inv.data(), f.LU.data(), swaps.data(), n, work.data());
    return inv;
}

/// @brief The unit-lower factor \f$L\f$, as a full matrix.
template <std::floating_point T>
inline mat<T> lower(const lu_result<T> &f) {
    const idx n = f.size();
    mat<T> L(n, n, T(0));
    for (idx i = 0; i < n; ++i) {
        for (idx j = 0; j < i; ++j) {
            L(i, j) = f.LU(i, j);
        }
        L(i, i) = T(1);
    }
    return L;
}

/// @brief The upper factor \f$U\f$, as a full matrix.
template <std::floating_point T>
inline mat<T> upper(const lu_result<T> &f) {
    const idx n = f.size();
    mat<T> U(n, n, T(0));
    for (idx i = 0; i < n; ++i) {
        for (idx j = i; j < n; ++j) {
            U(i, j) = f.LU(i, j);
        }
    }
    return U;
}

} // namespace num
