/// @file lu.hpp
/// @brief LU factorization with partial pivoting.
#pragma once

#include "core/debug.hpp"
#include "kernel/factor.hpp"
#include "kernel/raw.hpp"
#include "linear/matrix_utils.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include "container/parallel/lapack_wrapper.hpp"
#include "container/matrix.hpp"
#include "core/policy.hpp"
#include "linear/concepts.hpp"
#include "linear/matrix_properties.hpp"
#include <ostream>
#include <vector>

namespace num {

/// @brief Packed factorization \f$PA=LU\f$.
struct LUResult {
    Matrix LU;             ///< Packed unit-lower and upper factors.
    std::vector<idx> piv;  ///< Zero-based row swaps applied during factorization.
    bool singular = false; ///< True when a zero pivot was encountered.

    friend std::ostream &operator<<(std::ostream &os, const LUResult &r) {
        os << "LUResult{ dim: " << r.LU.rows() << "x" << r.LU.cols()
           << ", singular: " << (r.singular ? "true" : "false") << " }";
        return os;
    }
};

/// Factor a square matrix carrying a certified square dimension guarantee.
/// Factor with the sequential kernel.
LUResult lu(const linear::SquareMatrix<Matrix> &A, backend::seq_t);

/// Factor through LAPACK when it was configured; otherwise sequential.
LUResult lu(const linear::SquareMatrix<Matrix> &A, backend::lapack_t);

/// Any other tag has no distinct LU path and resolves to the sequential kernel.
template <class Tag>
inline LUResult lu(const linear::SquareMatrix<Matrix> &A, Tag) {
    return lu(A, backend::seq);
}

/// Factor with the strategy chosen by the build.
inline LUResult lu(const linear::SquareMatrix<Matrix> &A) {
    return lu(A, backend::factor);
}

/// Factor with a backend chosen at run time. A non-template overload, so it is
/// preferred over the tag template for an actual `Backend` value.
inline LUResult lu(const linear::SquareMatrix<Matrix> &A, Backend b) {
    return with_backend(b, [&](auto tag) { return lu(A, tag); });
}

namespace unsafe {

/// @brief Factor \f$PA = LU\f$ without requiring the square-dimension invariant.
template <class Tag = backend::factor_t>
inline LUResult lu(const Matrix &A, Tag tag = {}) {
    return num::lu(linear::SquareMatrix<Matrix>(A), tag);
}

} // namespace unsafe

/// @brief Rejects an untagged matrix at compile time.
template <class M, class Tag = backend::factor_t>
requires MatrixSpace<M> && (!SquareMatrixLike<M>)
LUResult lu(const M & /*untagged*/, Tag = {}) {
    static_assert(SquareMatrixLike<M>,
                  "lu() requires a matrix carrying the square-dimension invariant. "
                  "Establish it with num::assume_square(A) or num::make_square(A). "
                  "To bypass the invariant deliberately, call num::unsafe::lu(A).");
    return {};
}

/// @brief Solve \f$Ax=b\f$ from a precomputed \f$PA=LU\f$ factorization.
void lu_solve(const LUResult &f, const Vector &b, Vector &x);

/// @brief Solve \f$AX=B\f$ from a precomputed \f$PA=LU\f$ factorization.
void lu_solve(const LUResult &f, const Matrix &B, Matrix &X);

/// Solve A^T x=b from a precomputed PA=LU factorization.
void lu_solve_transpose(const LUResult &f, const Vector &b, Vector &x);
/// Solve A^T X=B for several right-hand sides.
void lu_solve_transpose(const LUResult &f, const Matrix &B, Matrix &X);

/// Replace one or more right-hand sides with the corresponding solutions.
void solve_in_place(const LUResult &f, Vector &right_hand_side);
void solve_in_place(const LUResult &f, Matrix &right_hand_sides);

/// @brief Compute \f$\det(A)=\det(P)^{-1}\prod_i U_{ii}\f$.
real lu_det(const LUResult &f);

/// @brief Compute \f$A^{-1}\f$ by solving \f$AX=I\f$.
Matrix lu_inv(const LUResult &f);



namespace backends {

namespace seq {
inline LUResult lu(const Matrix &A) {
    constexpr real singular_tol = 1e-14;
    const idx n = A.rows();
    LUResult f;
    f.LU = A;
    f.piv.resize(n);
    f.singular = false;

    Matrix &M = f.LU;
    std::vector<real> col_k(n);
    std::vector<real> lik_col(n);

    for (idx k = 0; k < n; ++k) {
        const idx len = n - k;
        for (idx i = 0; i < len; ++i) {
            col_k[i] = M(k + i, k);
        }

        const idx pivot_offset = kernel::raw::argmax_abs(col_k.data(), len);
        const idx pivot_row = k + pivot_offset;
        f.piv[k] = pivot_row;

        if (pivot_row != k) {
            kernel::raw::swap_rows(M.data(), n, k, pivot_row, n);
        }

        if (std::abs(M(k, k)) < singular_tol) {
            f.singular = true;
            continue;
        }

        const real inv_ukk = real(1) / M(k, k);
        for (idx i = k + 1; i < n; ++i) {
            M(i, k) *= inv_ukk;
            lik_col[i - (k + 1)] = M(i, k);
        }

        if (k + 1 < n) {
            kernel::raw::ger(&M(k + 1, k + 1), n, lik_col.data(), &M(k, k + 1), -1.0, n - 1 - k,
                             n - 1 - k);
        }
    }

    return f;
}
} // namespace seq

namespace lapack {
inline LUResult lu(const Matrix &A) {
#if defined(NUMERICS_HAS_LAPACK)
    const idx n = A.rows();
    LUResult f;
    f.LU = A;
    f.piv.resize(n);
    f.singular = false;

    std::vector<lapack_int> ipiv(n);
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
        f.piv[k] = static_cast<idx>(ipiv[k] - 1);
    }

    return f;
#else
    return seq::lu(A);
#endif
}
} // namespace lapack

} // namespace backends

inline LUResult lu(const linear::SquareMatrix<Matrix> &A, backend::seq_t) {
    return backends::seq::lu(A.base());
}

inline LUResult lu(const linear::SquareMatrix<Matrix> &A, backend::lapack_t) {
    return backends::lapack::lu(A.base());
}

inline void lu_solve(const LUResult &f, const Vector &b, Vector &x) {
    const idx n = f.LU.rows();
    const Matrix &M = f.LU;
    Vector y = b;

    for (idx k = 0; k < n; ++k) {
        if (f.piv[k] != k) {
            std::swap(y[k], y[f.piv[k]]);
        }
    }

    for (idx i = 1; i < n; ++i) {
        for (idx j = 0; j < i; ++j) {
            y[i] -= M(i, j) * y[j];
        }
    }

    for (idx i = n; i-- > 0;) {
        for (idx j = i + 1; j < n; ++j) {
            y[i] -= M(i, j) * y[j];
        }
        y[i] /= M(i, i);
    }

    x = std::move(y);
}

inline void lu_solve(const LUResult &f, const Matrix &B, Matrix &X) {
    const idx n = B.rows();
    if (f.LU.rows() != n || f.LU.cols() != n) {
        throw std::invalid_argument("lu_solve: dimension mismatch");
    }
    X = B;
#if defined(NUMERICS_HAS_LAPACK)
    std::vector<lapack_int> pivots(n);
    for (idx index = 0; index < n; ++index) {
        pivots[index] = static_cast<lapack_int>(f.piv[index] + 1);
    }
    const int info =
        LAPACKE_dgetrs(LAPACK_ROW_MAJOR, 'N', static_cast<lapack_int>(n),
                       static_cast<lapack_int>(B.cols()), f.LU.data(), static_cast<lapack_int>(n),
                       pivots.data(), X.data(), static_cast<lapack_int>(B.cols()));
    if (info != 0) {
        throw std::runtime_error("lu_solve: LAPACK block solve failed");
    }
#else
    for (idx k = 0; k < n; ++k) {
        if (f.piv[k] != k) {
            for (idx column = 0; column < B.cols(); ++column) {
                std::swap(X(k, column), X(f.piv[k], column));
            }
        }
    }
    for (idx row = 1; row < n; ++row) {
        for (idx k = 0; k < row; ++k) {
            for (idx column = 0; column < B.cols(); ++column) {
                X(row, column) -= f.LU(row, k) * X(k, column);
            }
        }
    }
    for (idx row = n; row-- > 0;) {
        for (idx k = row + 1; k < n; ++k) {
            for (idx column = 0; column < B.cols(); ++column) {
                X(row, column) -= f.LU(row, k) * X(k, column);
            }
        }
        for (idx column = 0; column < B.cols(); ++column) {
            X(row, column) /= f.LU(row, row);
        }
    }
#endif
}

inline void lu_solve_transpose(const LUResult &f, const Vector &b, Vector &x) {
    const idx n = f.LU.rows();
    if (f.LU.cols() != n || b.size() != n) {
        throw std::invalid_argument("lu_solve_transpose: dimension mismatch");
    }
    Vector work = b;
    // U^T q = b.
    for (idx row = 0; row < n; ++row) {
        for (idx column = 0; column < row; ++column) {
            work[row] -= f.LU(column, row) * work[column];
        }
        work[row] /= f.LU(row, row);
    }
    // L^T y = q; L has a unit diagonal.
    for (idx row = n; row-- > 0;) {
        for (idx column = row + 1; column < n; ++column) {
            work[row] -= f.LU(column, row) * work[column];
        }
    }
    // x = P^T y: undo row interchanges in reverse order.
    for (idx step = n; step-- > 0;) {
        if (f.piv[step] != step) {
            std::swap(work[step], work[f.piv[step]]);
        }
    }
    x = std::move(work);
}

inline void lu_solve_transpose(const LUResult &f, const Matrix &B, Matrix &X) {
    const idx n = f.LU.rows();
    if (f.LU.cols() != n || B.rows() != n) {
        throw std::invalid_argument("lu_solve_transpose: dimension mismatch");
    }
    X = Matrix(n, B.cols(), 0.0);
    Vector right_hand_side(n, 0.0);
    Vector solution(n, 0.0);
    for (idx column = 0; column < B.cols(); ++column) {
        for (idx row = 0; row < n; ++row) {
            right_hand_side[row] = B(row, column);
        }
        lu_solve_transpose(f, right_hand_side, solution);
        for (idx row = 0; row < n; ++row) {
            X(row, column) = solution[row];
        }
    }
}

inline void solve_in_place(const LUResult &f, Vector &right_hand_side) {
    Vector result(right_hand_side.size(), 0.0);
    lu_solve(f, right_hand_side, result);
    right_hand_side = std::move(result);
}

inline void solve_in_place(const LUResult &f, Matrix &right_hand_sides) {
    Matrix result;
    lu_solve(f, right_hand_sides, result);
    right_hand_sides = std::move(result);
}

inline real lu_det(const LUResult &f) {
    const idx n = f.LU.rows();
    real det = real(1);
    for (idx i = 0; i < n; ++i) {
        det *= f.LU(i, i);
    }
    idx swaps = 0;
    for (idx k = 0; k < n; ++k) {
        if (f.piv[k] != k) {
            ++swaps;
        }
    }
    return (swaps % 2 == 0) ? det : -det;
}

inline Matrix lu_inv(const LUResult &f) {
    const idx n = f.LU.rows();
    Matrix inv = f.LU;
#if defined(NUMERICS_HAS_LAPACK)
    std::vector<lapack_int> ipiv(n);
    for (idx i = 0; i < n; ++i) ipiv[i] = static_cast<lapack_int>(f.piv[i] + 1);
    LAPACKE_dgetri(LAPACK_ROW_MAJOR, static_cast<lapack_int>(n), inv.data(),
                   static_cast<lapack_int>(n), ipiv.data());
#else
    std::vector<real> work(n);
    kernel::raw::lu_invert(inv.data(), f.piv.data(), n, work.data());
#endif
    return inv;
}

} // namespace num
