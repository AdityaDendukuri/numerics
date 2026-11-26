/// @file linear/factorization/cholesky.hpp
/// @brief Dense Cholesky factorization.
#pragma once

#include "blas/matrix_ops.hpp"
#include "container/matrix.hpp"
#include "container/vector.hpp"
#include "core/debug.hpp"
#include "core/policy.hpp"
#include "kernel/kernel.hpp"
#include "lapack/lapack_wrapper.hpp"
#include "linear/matrix_properties.hpp"
#include "linear/solvers/solver_result.hpp"
#include "operator/concepts.hpp"
#include <cmath>
#include <ostream>
#include <stdexcept>
#include <vector>

namespace num {

/// @brief Factor a symmetric positive definite matrix by Cholesky: \f$A = L L^T\f$.
/// Reads the lower triangle; `L` and `A` may alias. Returns false at a non-positive pivot.
template <std::floating_point T>
[[nodiscard]] inline bool cholesky(T *L, const T *A, idx n) noexcept {
    for (idx i = 0; i < n; ++i) {
        for (idx j = 0; j <= i; ++j) {
            T sum = A[(i * n) + j];
            for (idx k = 0; k < j; ++k) {
                sum -= L[(i * n) + k] * L[(j * n) + k];
            }
            if (i == j) {
                if (!(sum > T(0))) {
                    return false;
                }
                L[(i * n) + j] = std::sqrt(sum);
            } else {
                L[(i * n) + j] = sum / L[(j * n) + j];
            }
        }
        for (idx j = i + 1; j < n; ++j) {
            L[(i * n) + j] = T(0);
        }
    }
    return true;
}

/// @brief Factor by Cholesky without blocking, in place: \f$A = L L^T\f$, zeros above the diagonal.
template <std::floating_point T>
[[nodiscard]] inline bool cholesky_block(T *A, idx lda, idx n) noexcept {
    for (idx i = 0; i < n; ++i) {
        for (idx j = 0; j <= i; ++j) {
            T sum = A[(i * lda) + j];
            for (idx p = 0; p < j; ++p) {
                sum -= A[(i * lda) + p] * A[(j * lda) + p];
            }
            if (i == j) {
                if (!(sum > T(0))) {
                    return false;
                }
                A[(i * lda) + j] = std::sqrt(sum);
            } else {
                A[(i * lda) + j] = sum / A[(j * lda) + j];
            }
        }
        for (idx j = i + 1; j < n; ++j) {
            A[(i * lda) + j] = T(0);
        }
    }
    return true;
}

/// @brief Factor by Cholesky by blocks, in place: \f$A = L L^T\f$.
template <std::floating_point T>
[[nodiscard]] inline bool cholesky_blocked(T *A, idx lda, idx n, idx block_size) noexcept {
    for (idx kk = 0; kk < n; kk += block_size) {
        const idx kb = std::min(block_size, n - kk);
        T *diagonal = A + (kk * lda) + kk;

        // L11 <- chol(A11), retaining the parent row stride.
        const bool ok = kb > kernel::detail::trsm_block
                            ? cholesky_blocked(diagonal, lda, kb, kernel::detail::trsm_block)
                            : cholesky_block(diagonal, lda, kb);
        if (!ok) {
            return false;
        }

        const idx trailing = n - (kk + kb);
        if (trailing == 0) {
            continue;
        }

        // L21 <- A21 * L11^{-T}.
        kernel::trsm_lower_transpose_right_inplace(A + ((kk + kb) * lda) + kk, lda, diagonal, lda,
                                                   trailing, kb);

        // A22 <- A22 - L21*L21^T (lower triangle only).
        kernel::syrk_lower(A + (((kk + kb) * lda) + (kk + kb)), lda, A + (((kk + kb) * lda) + kk),
                           lda, T(-1), T(1), trailing, kb);
    }
    return true;
}

/// @brief Factor by Cholesky by blocks, using level-3 BLAS: \f$A = L L^T\f$.
[[nodiscard]] inline bool cholesky_blocked_blas(real *A, idx lda, idx n,
                                                idx block_size = 256) noexcept {
    if (block_size == 0)
        block_size = 1;
    for (idx kk = 0; kk < n; kk += block_size) {
        const idx kb = std::min(block_size, n - kk);
        real *diagonal = A + (kk * lda) + kk;
        if (!cholesky_blocked(diagonal, lda, kb, kernel::detail::trsm_block))
            return false;

        const idx trailing = n - kk - kb;
        if (trailing == 0)
            continue;

        real *panel = A + ((kk + kb) * lda) + kk;
        constexpr idx row_block = 256;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (trailing >= 2 * row_block)
#endif
        for (idx row = 0; row < trailing; row += row_block) {
            const idx rows = std::min(row_block, trailing - row);
            blas::trsm_right_lower_transpose(diagonal, lda, panel + (row * lda), lda, rows, kb);
        }

        real *trailing_matrix = A + ((kk + kb) * lda) + kk + kb;
        constexpr idx column_block = 256;
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1) if (trailing >= 2 * column_block)
#endif
        for (idx column = 0; column < trailing; column += column_block) {
            const idx columns = std::min(column_block, trailing - column);
            blas::syrk_lower(-1.0, panel + (column * lda), lda, 1.0,
                             trailing_matrix + (column * lda) + column, lda, columns, kb);
            const idx below = trailing - column - columns;
            if (below > 0) {
                blas::gemm(-1.0, panel + ((column + columns) * lda), lda, false,
                           panel + (column * lda), lda, true, 1.0,
                           trailing_matrix + ((column + columns) * lda) + column, lda, below,
                           columns, kb);
            }
        }
    }
    return true;
}

/// @brief Factor by Cholesky by blocks, in place: \f$A \leftarrow L\f$ with \f$A = L L^T\f$.
template <std::floating_point T>
[[nodiscard]] inline bool cholesky_blocked(T *A, idx n, idx block_size = 256) noexcept {
    if (n == 0) {
        return true;
    }
    if (block_size == 0) {
        block_size = 1;
    }
    if (!num::cholesky_blocked(A, n, n, block_size)) {
        return false;
    }
    // The packed lower factor is the public result; make the unused triangle explicit.
    for (idx i = 0; i < n; ++i) {
        for (idx j = i + 1; j < n; ++j) {
            A[(i * n) + j] = T(0);
        }
    }
    return true;
}

/// @brief Solve with a Cholesky factor: \f$x \leftarrow L^{-T} L^{-1} b\f$. `x` and `b` may alias.
template <std::floating_point T>
inline void cholesky_solve(T *x, const T *L, const T *b, idx n) noexcept {
    // The public contract permits x == b, so select the explicitly alias-safe
    // raw path rather than making a false restrict promise to the compiler.
    kernel::trsv_lower(kernel::contract::alias_safe, x, L, b, n);
    kernel::trsv_transpose_lower(x, L, n, n);
}

} // namespace num

namespace num {

namespace linear {

/// @brief Test whether a matrix is symmetric within `tol` and has a Cholesky factorization.
[[nodiscard]] inline bool is_spd(const mat<real> &A, real tol = 1e-12) {
    if (!is_symmetric(A, tol)) {
        return false;
    }
    mat<real> factor(A.rows(), A.cols(), 0.0);
    return num::cholesky(factor.data(), A.data(), A.rows());
}

/// @brief Verify by Cholesky that a matrix is symmetric positive definite, then tag it `law::spd`.
/// @throws std::invalid_argument If `A` is not symmetric within `tol` or not positive definite.
template <class Mat = mat<real>>
[[nodiscard]] inline with_law<Mat, law::spd> make_spd(Mat A, real tol = 1e-12) {
    if (!is_spd(A, tol)) {
        throw std::invalid_argument("make_spd: matrix is not symmetric positive definite");
    }
    return with_law<Mat, law::spd>(std::move(A));
}

} // namespace linear

using linear::make_spd;

/// @brief Result of a Cholesky factorization: \f$A = L L^T\f$.
struct cholesky_result {
    mat<real> L;                    ///< Lower-triangular factor, valid when `positive_definite`.
    bool positive_definite = false; ///< False when a pivot was not positive.

    /// @brief Return the order \f$n\f$.
    [[nodiscard]] idx size() const noexcept { return L.rows(); }

    friend std::ostream &operator<<(std::ostream &os, const cholesky_result &r) {
        os << "cholesky_result{ dim: " << r.L.rows() << "x" << r.L.cols()
           << ", positive_definite: " << (r.positive_definite ? "true" : "false") << " }";
        return os;
    }
};

namespace detail {
cholesky_result cholesky_impl(const mat<real> &A);
}

/// @brief Factor a matrix tagged SPD by Cholesky: \f$A = L L^T\f$.
cholesky_result cholesky(const with_law<mat<real>, law::spd> &A);

namespace unsafe {

/// @brief Factor by Cholesky without checking that the matrix is SPD: \f$A = L L^T\f$.
/// Failure is reported in `positive_definite`.
/// @warning Positive definiteness is taken on faith; prefer tagging `A` with `assume_spd`.
cholesky_result cholesky(const mat<real> &A);

} // namespace unsafe

/// @brief Reject an untagged matrix at compile time; tag it SPD first.
template <class M>
requires matrix_space<M> && (!claims<M, law::spd>)
cholesky_result cholesky(const M & /*untagged*/) {
    static_assert(claims<M, law::spd>,
                  "cholesky() requires a matrix carrying the SPD invariant. "
                  "Establish it with num::assume_spd(A) (asserted, sampled at runtime) or "
                  "num::make_spd(A) (verified exhaustively). "
                  "To bypass the invariant deliberately, call num::unsafe::cholesky(A).");
    return {};
}

/// @brief Solve with a Cholesky factor: \f$x \leftarrow A^{-1} b\f$. `x` may alias `b`.
void solve(const cholesky_result &f, const vec<real> &b, vec<real> &x);
/// @brief Solve with a Cholesky factor: \f$X \leftarrow A^{-1} B\f$. `X` may alias `B`.
void solve(const cholesky_result &f, const mat<real> &B, mat<real> &X);
/// @brief Solve the transposed system with a Cholesky factor:
/// \f$x \leftarrow A^{-T} b = A^{-1} b\f$.
void solve_transpose(const cholesky_result &f, const vec<real> &b, vec<real> &x);
void solve_transpose(const cholesky_result &f, const mat<real> &B, mat<real> &X);

/// @brief Update a Cholesky factor for a rank-1 change: \f$A \leftarrow A + x x^T\f$, \f$O(n^2)\f$.
void cholesky_update(cholesky_result &factor, const vec<real> &update);

/// @brief Downdate a Cholesky factor for a rank-1 change:
/// \f$A \leftarrow A - x x^T\f$, \f$O(n^2)\f$.
/// @throws std::runtime_error If the result is not positive definite.
void cholesky_downdate(cholesky_result &factor, const vec<real> &update);

namespace lapack {

/// @brief Factor by Cholesky through LAPACK `dpotrf`: \f$A = L L^T\f$.
/// @note Not the default: the in-tree blocked kernel measured faster at every size tried.
inline cholesky_result cholesky(const mat<real> &A) {
    if (A.rows() != A.cols()) {
        throw std::invalid_argument("cholesky: matrix must be square");
    }
#if defined(NUMERICS_HAS_LAPACK)
    const idx n = A.rows();
    mat<real> L = A;
    int info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', static_cast<lapack_int>(n), L.data(),
                              static_cast<lapack_int>(n));
    if (info != 0) {
        return {std::move(L), false};
    }
    for (idx i = 0; i < n; ++i) {
        for (idx j = i + 1; j < n; ++j) {
            L(i, j) = 0.0;
        }
    }
    return {std::move(L), true};
#else
    mat<real> L = A;
    const bool ok = num::cholesky_blocked(L.data(), A.rows());
    return {std::move(L), ok};
#endif
}

/// @brief Solve with a Cholesky factor through LAPACK `dpotrs`:
/// \f$X \leftarrow A^{-1} B\f$. `X` may alias `B`.
inline void solve(const cholesky_result &f, const mat<real> &B, mat<real> &X) {
#if defined(NUMERICS_HAS_LAPACK)
    const idx n = f.L.rows();
    if (!f.positive_definite) {
        throw std::invalid_argument("cholesky solve: the matrix is not positive definite");
    }
    if (B.rows() != n) {
        throw std::invalid_argument("cholesky solve: dimension mismatch");
    }
    if (&X != &B) {
        X = B;
    }
    LAPACKE_dpotrs(LAPACK_ROW_MAJOR, 'L', static_cast<lapack_int>(n),
                   static_cast<lapack_int>(X.cols()), f.L.data(), static_cast<lapack_int>(n),
                   X.data(), static_cast<lapack_int>(X.cols()));
#else
    num::solve(f, B, X);
#endif
}

/// @brief Solve with a Cholesky factor through LAPACK `dpotrs`:
/// \f$x \leftarrow A^{-1} b\f$. `x` may alias `b`.
inline void solve(const cholesky_result &f, const vec<real> &b, vec<real> &x) {
#if defined(NUMERICS_HAS_LAPACK)
    const idx n = f.L.rows();
    if (!f.positive_definite) {
        throw std::invalid_argument("cholesky solve: the matrix is not positive definite");
    }
    if (b.size() != n) {
        throw std::invalid_argument("cholesky solve: dimension mismatch");
    }
    if (&x != &b) {
        x = b;
    }
    LAPACKE_dpotrs(LAPACK_ROW_MAJOR, 'L', static_cast<lapack_int>(n), 1, f.L.data(),
                   static_cast<lapack_int>(n), x.data(), 1);
#else
    num::solve(f, b, x);
#endif
}

} // namespace lapack

namespace detail {

inline cholesky_result cholesky_impl(const mat<real> &A) {
    if (A.rows() != A.cols()) {
        throw std::invalid_argument("cholesky: matrix must be square");
    }
    mat<real> L = A;
    constexpr idx blas_threshold = 256;
    const bool ok = A.rows() >= blas_threshold
                        ? num::cholesky_blocked_blas(L.data(), A.rows(), A.rows())
                        : num::cholesky_blocked(L.data(), A.rows());
    for (idx i = 0; i < A.rows(); ++i)
        for (idx j = i + 1; j < A.rows(); ++j)
            L(i, j) = 0.0;
    return {std::move(L), ok};
}

} // namespace detail

inline cholesky_result cholesky(const with_law<mat<real>, law::spd> &A) {
    return detail::cholesky_impl(A.base());
}

namespace unsafe {

inline cholesky_result cholesky(const mat<real> &A) {
    return detail::cholesky_impl(A);
}

} // namespace unsafe

namespace detail {
inline void check_rhs(const cholesky_result &f, idx rows) {
    if (!f.positive_definite) {
        throw std::invalid_argument("cholesky solve: the matrix is not positive definite");
    }
    if (f.L.cols() != f.L.rows() || rows != f.L.rows()) {
        throw std::invalid_argument("cholesky solve: dimension mismatch");
    }
}
} // namespace detail

inline void solve(const cholesky_result &f, const vec<real> &b, vec<real> &x) {
    detail::check_rhs(f, b.size());
    const idx n = f.L.rows();
    x = b;
    kernel::trsv_lower_inplace(x.data(), f.L.data(), n);
    kernel::trsv_transpose_lower(x.data(), f.L.data(), n, n);
}

inline void solve(const cholesky_result &f, const mat<real> &B, mat<real> &X) {
    detail::check_rhs(f, B.rows());
    const idx n = f.L.rows();

    X = B;
    constexpr idx blas_threshold = 256;
    if (n < blas_threshold) {
        kernel::trsm_lower_inplace(X.data(), B.cols(), f.L.data(), n, B.cols());
        kernel::trsm_lower_transpose_inplace(X.data(), B.cols(), f.L.data(), n, B.cols());
        return;
    }
    constexpr idx rhs_block = 32;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (B.cols() >= 2 * rhs_block)
#endif
    for (idx column = 0; column < B.cols(); column += rhs_block) {
        const idx columns = std::min(rhs_block, B.cols() - column);
        blas::trsm_left(f.L.data(), n, X.data() + column, B.cols(), n, columns, false, false,
                        false);
        blas::trsm_left(f.L.data(), n, X.data() + column, B.cols(), n, columns, false, true,
                        false);
    }
}

inline void solve_transpose(const cholesky_result &f, const vec<real> &b, vec<real> &x) {
    solve(f, b, x);
}

inline void solve_transpose(const cholesky_result &f, const mat<real> &B, mat<real> &X) {
    solve(f, B, X);
}

inline void cholesky_update(cholesky_result &factor, const vec<real> &update) {
    if (!factor.positive_definite || factor.L.rows() != update.size()) {
        throw std::invalid_argument("cholesky_update: invalid factor or update size");
    }
    vec<real> work = update;
    for (idx column = 0; column < work.size(); ++column) {
        const real diagonal = factor.L(column, column);
        const real replacement = std::hypot(diagonal, work[column]);
        const real cosine = replacement / diagonal;
        const real sine = work[column] / diagonal;
        factor.L(column, column) = replacement;
        for (idx row = column + 1; row < work.size(); ++row) {
            factor.L(row, column) = (factor.L(row, column) + (sine * work[row])) / cosine;
            work[row] = (cosine * work[row]) - (sine * factor.L(row, column));
        }
    }
}

inline void cholesky_downdate(cholesky_result &factor, const vec<real> &update) {
    if (!factor.positive_definite || factor.L.rows() != update.size()) {
        throw std::invalid_argument("cholesky_downdate: invalid factor or update size");
    }
    mat<real> candidate = factor.L;
    vec<real> work = update;
    for (idx column = 0; column < work.size(); ++column) {
        const real diagonal = candidate(column, column);
        const real square = (diagonal * diagonal) - (work[column] * work[column]);
        if (!(square > 0.0)) {
            throw std::domain_error("cholesky_downdate: result is not positive definite");
        }
        const real replacement = std::sqrt(square);
        const real cosine = replacement / diagonal;
        const real sine = work[column] / diagonal;
        candidate(column, column) = replacement;
        for (idx row = column + 1; row < work.size(); ++row) {
            candidate(row, column) = (candidate(row, column) - (sine * work[row])) / cosine;
            work[row] = (cosine * work[row]) - (sine * candidate(row, column));
        }
    }
    factor.L = std::move(candidate);
}

} // namespace num
