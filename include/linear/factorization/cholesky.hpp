/// @file linear/factorization/cholesky.hpp
/// @brief Dense Cholesky factorization for SPD matrices.
#pragma once

#include "kernel/factor.hpp"
#include "core/debug.hpp"
#include "kernel/raw.hpp"
#include <cmath>
#include <stdexcept>
#include <vector>
#include "container/parallel/lapack_wrapper.hpp"
#include "container/matrix.hpp"
#include "container/vector.hpp"
#include "linear/concepts.hpp"
#include <ostream>

namespace num {

/// @brief Lower-triangular factorization \f$A=LL^T\f$.
struct CholeskyResult {
    Matrix L;             ///< Lower-triangular factor when successful.
    bool success = false; ///< False when the input is not positive definite.

    friend std::ostream &operator<<(std::ostream &os, const CholeskyResult &r) {
        os << "CholeskyResult{ dim: " << r.L.rows() << "x" << r.L.cols()
           << ", success: " << (r.success ? "true" : "false") << " }";
        return os;
    }
};

namespace detail {
CholeskyResult cholesky_impl(const Matrix &A);
}

/// Factor a matrix whose SPD property has already been established.
CholeskyResult cholesky(const linear::SPDMatrix<Matrix> &A);

namespace unsafe {

/// @brief Factor \f$A = LL^T\f$ without requiring or checking the SPD invariant.
///
/// The deliberate escape hatch. Calling this says, at the call site and in a form
/// that survives grep, that the SPD precondition is being taken on faith. Nothing
/// is sampled; if A is not positive definite the factorization simply reports
/// failure through `CholeskyResult::success`.
CholeskyResult cholesky(const Matrix &A);

} // namespace unsafe

/// @brief Rejects an untagged matrix at compile time.
///
/// Cholesky is defined only for symmetric positive definite matrices, so a raw
/// `Matrix` does not satisfy its precondition. This overload exists to say so in a
/// diagnostic rather than to run: a warning can be silenced by an unrelated
/// `-Wno-` flag, whereas this cannot compile.
template <class M>
requires MatrixSpace<M> && (!SPDMatrixLike<M>)
CholeskyResult cholesky(const M & /*untagged*/) {
    static_assert(SPDMatrixLike<M>,
                  "cholesky() requires a matrix carrying the SPD invariant. "
                  "Establish it with num::assume_spd(A) (asserted, sampled at runtime) or "
                  "num::make_spd(A) (verified exhaustively). "
                  "To bypass the invariant deliberately, call num::unsafe::cholesky(A).");
    return {};
}

/// Solve Ax=b from a reusable Cholesky factorization.
void cholesky_solve(const CholeskyResult &f, const Vector &b, Vector &x);

/// @brief Solve \f$AX=B\f$ for several right-hand sides at once.
void cholesky_solve(const CholeskyResult &f, const Matrix &B, Matrix &X);

/// Replace one or more right-hand sides with the corresponding solutions.
void solve_in_place(const CholeskyResult &f, Vector &right_hand_side);
void solve_in_place(const CholeskyResult &f, Matrix &right_hand_sides);

/// Replace A=LL^T by A+x*x^T in O(n^2).
void cholesky_update(CholeskyResult &factor, const Vector &update);

/// Replace A=LL^T by A-x*x^T in O(n^2), or throw if it is not SPD.
void cholesky_downdate(CholeskyResult &factor, const Vector &update);




namespace detail {

inline CholeskyResult cholesky_impl(const Matrix &A) {
    if (A.rows() != A.cols()) {
        throw std::invalid_argument("cholesky: matrix must be square");
    }

    const idx n = A.rows();

#if defined(NUMERICS_HAS_LAPACK)
    Matrix L = A;
    int info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', static_cast<lapack_int>(n), L.data(),
                              static_cast<lapack_int>(n));
    if (info != 0) {
        return {std::move(L), false};
    }

    // Zero out upper triangle for lower triangular result L
    for (idx i = 0; i < n; ++i) {
        for (idx j = i + 1; j < n; ++j) {
            L(i, j) = 0.0;
        }
    }

    return {std::move(L), true};
#else
    // The sequential factorization itself lives in kernel/factor.hpp as a
    // raw-pointer kernel, so consuming projects can use it without this container.
    Matrix L = A;
    // A = L*L^T; blocked panels lower the cost of the trailing update.
    const bool ok = kernel::raw::cholesky_blocked(L.data(), n);
    return {std::move(L), ok};
#endif
}

} // namespace detail

inline CholeskyResult cholesky(const linear::SPDMatrix<Matrix> &A) {
    return detail::cholesky_impl(A.base());
}

namespace unsafe {

inline CholeskyResult cholesky(const Matrix &A) {
    return detail::cholesky_impl(A);
}

} // namespace unsafe

inline void cholesky_solve(const CholeskyResult &f, const Vector &b, Vector &x) {
    if (!f.success) {
        throw std::invalid_argument("cholesky_solve: factorization failed");
    }
    const idx n = f.L.rows();
    if (f.L.cols() != n || b.size() != n || x.size() != n) {
        throw std::invalid_argument("cholesky_solve: dimension mismatch");
    }

    x = b;
#if defined(NUMERICS_HAS_LAPACK)
    const int info = LAPACKE_dpotrs(LAPACK_ROW_MAJOR, 'L', static_cast<lapack_int>(n), 1,
                                    f.L.data(), static_cast<lapack_int>(n), x.data(), 1);
    if (info != 0) {
        throw std::runtime_error("cholesky_solve: LAPACK solve failed");
    }
#else
    kernel::raw::trsv_lower(x.data(), f.L.data(), b.data(), n);
    kernel::raw::trsv_transpose_lower(x.data(), f.L.data(), n, n);
#endif
}

inline void cholesky_solve(const CholeskyResult &f, const Matrix &B, Matrix &X) {
    if (!f.success) {
        throw std::invalid_argument("cholesky_solve: factorization failed");
    }
    const idx n = f.L.rows();
    if (f.L.cols() != n || B.rows() != n) {
        throw std::invalid_argument("cholesky_solve: dimension mismatch");
    }

    X = B;
#if defined(NUMERICS_HAS_LAPACK)
    const int info = LAPACKE_dpotrs(
        LAPACK_ROW_MAJOR, 'L', static_cast<lapack_int>(n), static_cast<lapack_int>(B.cols()),
        f.L.data(), static_cast<lapack_int>(n), X.data(), static_cast<lapack_int>(B.cols()));
    if (info != 0) {
        throw std::runtime_error("cholesky_solve: LAPACK block solve failed");
    }
#else
    // Y <- L^{-1}B, then X <- L^{-T}Y.
    kernel::raw::trsm_lower_inplace(X.data(), B.cols(), f.L.data(), n, B.cols());
    kernel::raw::trsm_lower_transpose_inplace(X.data(), B.cols(), f.L.data(), n, B.cols());
#endif
}

inline void solve_in_place(const CholeskyResult &f, Vector &right_hand_side) {
    Vector result(right_hand_side.size(), 0.0);
    cholesky_solve(f, right_hand_side, result);
    right_hand_side = std::move(result);
}

inline void solve_in_place(const CholeskyResult &f, Matrix &right_hand_sides) {
    Matrix result;
    cholesky_solve(f, right_hand_sides, result);
    right_hand_sides = std::move(result);
}

inline void cholesky_update(CholeskyResult &factor, const Vector &update) {
    if (!factor.success || factor.L.rows() != update.size()) {
        throw std::invalid_argument("cholesky_update: invalid factor or update size");
    }
    Vector work = update;
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

inline void cholesky_downdate(CholeskyResult &factor, const Vector &update) {
    if (!factor.success || factor.L.rows() != update.size()) {
        throw std::invalid_argument("cholesky_downdate: invalid factor or update size");
    }
    Matrix candidate = factor.L;
    Vector work = update;
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
