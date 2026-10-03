/// @file linear/factorization/cholesky.hpp
/// @brief Dense Cholesky factorization for SPD matrices.
#pragma once

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

/// @brief Cholesky factorization \f$A = L L^T\f$ for symmetric positive definite \f$A\f$.
///
/// Writes L and zeroes the strict upper triangle; `L` and `A` may alias. Returns false at the
/// first non-positive pivot, which proves A is not positive definite.
///
/// @param L Output lower triangular factor, n*n.
/// @param A Input symmetric matrix, n*n; only the lower triangle is read.
/// @param n mat dimension.
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

/// @brief Unblocked lower Cholesky of an `n x n` block with row stride `lda`,
/// writing zeros above the diagonal.
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

/// @brief Blocked lower Cholesky of an `n x n` block with row stride `lda`.
///
/// Diagonal blocks wider than the `trsm`/`syrk` block are factored by
/// recursion, so the panel solve and trailing update always see blocks long
/// enough for their internal `gemm` to matter.
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

/// @brief In-place blocked Cholesky factorization, `A <- L` with `A = L*L^T`.
///
/// Panels of `block_size` columns; the panel solve and trailing update are the blocked `trsm`
/// and `syrk`, so large sizes run at `gemm` speed.
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

/// @brief Solve \f$A x = b\f$ from a Cholesky factor, via \f$L y = b\f$ then \f$L^T x = y\f$.
///
/// `x` and `b` may alias.
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

/// Test symmetry and positive definiteness by a Cholesky factorization, which fails exactly
/// when a pivot is not positive. The raw-pointer overload is used because the `mat` overload of
/// `num::cholesky` requires the invariant being tested.
[[nodiscard]] inline bool is_spd(const mat<real> &A, real tol = 1e-12) {
    if (!is_symmetric(A, tol)) {
        return false;
    }
    mat<real> factor(A.rows(), A.cols(), 0.0);
    return num::cholesky(factor.data(), A.data(), A.rows());
}

/// @brief Check symmetric positive definiteness exhaustively by a Cholesky factorization, then
/// attach it.
/// @throws std::invalid_argument If `A` is not symmetric within `tol`, or a pivot is not positive.
template <class Mat = mat<real>>
[[nodiscard]] inline with_law<Mat, law::spd> make_spd(Mat A, real tol = 1e-12) {
    if (!is_spd(A, tol)) {
        throw std::invalid_argument("make_spd: matrix is not symmetric positive definite");
    }
    return with_law<Mat, law::spd>(std::move(A));
}

} // namespace linear

using linear::make_spd;

/// @brief Lower-triangular factorization \f$A=LL^T\f$.
struct cholesky_result {
    mat<real> L;                    ///< Lower-triangular factor, valid when `positive_definite`.
    bool positive_definite = false; ///< False when a pivot was not positive.

    friend std::ostream &operator<<(std::ostream &os, const cholesky_result &r) {
        os << "cholesky_result{ dim: " << r.L.rows() << "x" << r.L.cols()
           << ", positive_definite: " << (r.positive_definite ? "true" : "false") << " }";
        return os;
    }
};

namespace detail {
cholesky_result cholesky_impl(const mat<real> &A);
}

/// Factor a matrix whose SPD property has already been established.
cholesky_result cholesky(const with_law<mat<real>, law::spd> &A);

namespace unsafe {

/// @brief Factor \f$A = LL^T\f$ without requiring or checking the SPD invariant.
///
/// The deliberate escape hatch. Calling this says, at the call site and in a form
/// that survives grep, that the SPD precondition is being taken on faith. Nothing
/// is sampled; if A is not positive definite the factorization simply reports
/// failure through `cholesky_result::positive_definite`.
cholesky_result cholesky(const mat<real> &A);

} // namespace unsafe

/// @brief Rejects an untagged matrix at compile time.
///
/// Cholesky is defined only for symmetric positive definite matrices, so a raw
/// `mat<real>` does not satisfy its precondition. This overload exists to say so in a
/// diagnostic rather than to run: a warning can be silenced by an unrelated
/// `-Wno-` flag, whereas this cannot compile.
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

/// @brief Solve \f$Ax=b\f$. `x` may be `b`.
void solve(const cholesky_result &f, const vec<real> &b, vec<real> &x);
/// @brief Solve \f$AX=B\f$. `X` may be `B`.
void solve(const cholesky_result &f, const mat<real> &B, mat<real> &X);
/// @brief Solve \f$A^Tx=b\f$, the same solve since \f$A\f$ is symmetric.
void solve_transpose(const cholesky_result &f, const vec<real> &b, vec<real> &x);
void solve_transpose(const cholesky_result &f, const mat<real> &B, mat<real> &X);

/// Replace A=LL^T by A+x*x^T in O(n^2).
void cholesky_update(cholesky_result &factor, const vec<real> &update);

/// Replace A=LL^T by A-x*x^T in O(n^2), or throw if it is not SPD.
void cholesky_downdate(cholesky_result &factor, const vec<real> &update);

namespace lapack {

/// Cholesky through LAPACKE's dpotrf. Not the default: the kernel's blocked
/// factorization measured faster at every size tried (see
/// `lapack_factor_threshold`), but the binding stays callable by name.
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

} // namespace lapack

namespace detail {

inline cholesky_result cholesky_impl(const mat<real> &A) {
    if (A.rows() != A.cols()) {
        throw std::invalid_argument("cholesky: matrix must be square");
    }
    // The blocked raw-pointer Cholesky above: panel solve and trailing update
    // through the packed trsm/syrk, faster than dpotrf through LAPACKE at every
    // size measured.
    mat<real> L = A;
    const bool ok = num::cholesky_blocked(L.data(), A.rows());
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
    // Y <- L^{-1}B, then X <- L^{-T}Y, both blocked over gemm. Never dpotrs:
    // the kernel solve is faster at every size, and on a threaded BLAS the
    // LAPACK call pays a thread wake-up per solve.
    kernel::trsm_lower_inplace(X.data(), B.cols(), f.L.data(), n, B.cols());
    kernel::trsm_lower_transpose_inplace(X.data(), B.cols(), f.L.data(), n, B.cols());
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
