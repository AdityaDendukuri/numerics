/// @file lu.hpp
/// @brief LU factorization with partial pivoting.
#pragma once

#include "core/types.hpp"
#include "core/debug.hpp"
#include "kernel/factor.hpp"
#include "kernel/kernel.hpp"
#include "linear/matrix_utils.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include "lapack/lapack_wrapper.hpp"
#include "container/matrix.hpp"
#include "core/policy.hpp"
#include "linear/concepts.hpp"
#include "linear/matrix_properties.hpp"
#include <ostream>
#include <vector>

namespace num {

/// @brief Packed factorization \f$PA=LU\f$.
struct lu_result {
    mat LU;             ///< Packed unit-lower and upper factors.
    array<idx> piv;  ///< Zero-based row swaps applied during factorization.
    bool singular = false; ///< True when a zero pivot was encountered.

    friend std::ostream &operator<<(std::ostream &os, const lu_result &r) {
        os << "lu_result{ dim: " << r.LU.rows() << "x" << r.LU.cols()
           << ", singular: " << (r.singular ? "true" : "false") << " }";
        return os;
    }
};

/// Factor a square matrix carrying a certified square dimension guarantee.
/// Picks the best available implementation at compile time: LAPACK (`dgetrf`)
/// if configured, else the in-tree sequential kernel. To force one explicitly,
/// call `num::lapack::lu`/`num::seq::lu` directly.
/// @return `lu_result`: `.LU` (packed factors), `.piv` (row pivots), `.singular`.
inline lu_result lu(const linear::sq_mat<mat> &A);

namespace unsafe {

/// @brief Factor \f$PA = LU\f$ without requiring the square-dimension invariant.
/// @return `lu_result`: `.LU` (packed factors), `.piv` (row pivots), `.singular`.
inline lu_result lu(const mat &A) { return num::lu(linear::sq_mat<mat>(A)); }

} // namespace unsafe

/// @brief Rejects an untagged matrix at compile time.
template <class M>
requires matrix_space<M> && (!square_matrix_like<M>)
lu_result lu(const M & /*untagged*/) {
    static_assert(square_matrix_like<M>,
                  "lu() requires a matrix carrying the square-dimension invariant. "
                  "Establish it with num::assume_square(A) or num::make_square(A). "
                  "To bypass the invariant deliberately, call num::unsafe::lu(A).");
    return {};
}

/// @brief Solve \f$Ax=b\f$ from a precomputed \f$PA=LU\f$ factorization.
void lu_solve(const lu_result &f, const vec &b, vec &x);

/// @brief Solve \f$AX=B\f$ from a precomputed \f$PA=LU\f$ factorization.
void lu_solve(const lu_result &f, const mat &B, mat &X);

/// Solve A^T x=b from a precomputed PA=LU factorization.
void lu_solve_transpose(const lu_result &f, const vec &b, vec &x);
/// Solve A^T X=B for several right-hand sides.
void lu_solve_transpose(const lu_result &f, const mat &B, mat &X);

/// Replace one or more right-hand sides with the corresponding solutions.
void solve_in_place(const lu_result &f, vec &right_hand_side);
void solve_in_place(const lu_result &f, mat &right_hand_sides);

/// @brief Compute \f$\det(A)=\det(P)^{-1}\prod_i U_{ii}\f$.
real lu_det(const lu_result &f);

/// @brief Compute \f$A^{-1}\f$ by solving \f$AX=I\f$.
mat lu_inv(const lu_result &f);



namespace seq {
inline lu_result lu(const mat &A) {
    constexpr real singular_tol = 1e-14;
    const idx n = A.rows();
    lu_result f;
    f.LU = A;
    f.piv.resize(n);
    f.singular = false;

    mat &M = f.LU;
    array<real> col_k(n);
    array<real> lik_col(n);

    for (idx k = 0; k < n; ++k) {
        const idx len = n - k;
        for (idx i = 0; i < len; ++i) {
            col_k[i] = M(k + i, k);
        }

        const idx pivot_offset = kernel::argmax_abs(col_k.data(), len);
        const idx pivot_row = k + pivot_offset;
        f.piv[k] = pivot_row;

        if (pivot_row != k) {
            kernel::swap_rows(M.data(), n, k, pivot_row, n);
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
            kernel::ger(&M(k + 1, k + 1), n, lik_col.data(), &M(k, k + 1), -1.0, n - 1 - k,
                             n - 1 - k);
        }
    }

    return f;
}
} // namespace seq

namespace lapack {
inline lu_result lu(const mat &A) {
#if defined(NUMERICS_HAS_LAPACK)
    const idx n = A.rows();
    lu_result f;
    f.LU = A;
    f.piv.resize(n);
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
        f.piv[k] = static_cast<idx>(ipiv[k] - 1);
    }

    return f;
#else
    return seq::lu(A);
#endif
}
} // namespace lapack

inline lu_result lu(const linear::sq_mat<mat> &A) {
#if defined(NUMERICS_HAS_LAPACK)
    return lapack::lu(A.base());
#else
    return seq::lu(A.base());
#endif
}

inline void lu_solve(const lu_result &f, const vec &b, vec &x) {
    const idx n = f.LU.rows();
    const mat &M = f.LU;
    vec y = b;

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

inline void lu_solve(const lu_result &f, const mat &B, mat &X) {
    const idx n = B.rows();
    if (f.LU.rows() != n || f.LU.cols() != n) {
        throw std::invalid_argument("lu_solve: dimension mismatch");
    }
    X = B;
#if defined(NUMERICS_HAS_LAPACK)
    array<lapack_int> pivots(n);
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

inline void lu_solve_transpose(const lu_result &f, const vec &b, vec &x) {
    const idx n = f.LU.rows();
    if (f.LU.cols() != n || b.size() != n) {
        throw std::invalid_argument("lu_solve_transpose: dimension mismatch");
    }
    vec work = b;
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

inline void lu_solve_transpose(const lu_result &f, const mat &B, mat &X) {
    const idx n = f.LU.rows();
    if (f.LU.cols() != n || B.rows() != n) {
        throw std::invalid_argument("lu_solve_transpose: dimension mismatch");
    }
    X = mat(n, B.cols(), 0.0);
    vec right_hand_side(n, 0.0);
    vec solution(n, 0.0);
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

inline void solve_in_place(const lu_result &f, vec &right_hand_side) {
    vec result(right_hand_side.size(), 0.0);
    lu_solve(f, right_hand_side, result);
    right_hand_side = std::move(result);
}

inline void solve_in_place(const lu_result &f, mat &right_hand_sides) {
    mat result;
    lu_solve(f, right_hand_sides, result);
    right_hand_sides = std::move(result);
}

inline real lu_det(const lu_result &f) {
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

inline mat lu_inv(const lu_result &f) {
    const idx n = f.LU.rows();
    mat inv = f.LU;
#if defined(NUMERICS_HAS_LAPACK)
    array<lapack_int> ipiv(n);
    for (idx i = 0; i < n; ++i) ipiv[i] = static_cast<lapack_int>(f.piv[i] + 1);
    LAPACKE_dgetri(LAPACK_ROW_MAJOR, static_cast<lapack_int>(n), inv.data(),
                   static_cast<lapack_int>(n), ipiv.data());
#else
    array<real> work(n);
    kernel::lu_invert(inv.data(), f.piv.data(), n, work.data());
#endif
    return inv;
}

} // namespace num
