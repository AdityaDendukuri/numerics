/// @file lu.hpp
/// @brief LU factorization with partial pivoting.
#pragma once

#include "container/matrix.hpp"
#include "core/debug.hpp"
#include "core/policy.hpp"
#include "core/types.hpp"
#include "kernel/factor.hpp"
#include "kernel/kernel.hpp"
#include "lapack/lapack_wrapper.hpp"
#include "linear/concepts.hpp"
#include "linear/matrix_properties.hpp"
#include "linear/matrix_utils.hpp"
#include <algorithm>
#include <cmath>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace num {

/// @brief Packed factorization \f$PA=LU\f$.
struct lu_result {
    mat LU;                ///< Packed unit-lower and upper factors.
    array<idx> piv;        ///< Zero-based row swaps applied during factorization.
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
inline lu_result lu(const mat &A) {
    return num::lu(linear::sq_mat<mat>(A));
}

} // namespace unsafe

/// @brief Rejects an untagged matrix at compile time.
template <class M>
    requires matrix_space<M> && (!square_matrix_like<M>)lu_result lu(const M & /*untagged*/) {
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
/// Blocked partial-pivoting LU through `kernel::lu_factor_blocked`: panel
/// factorization, then a `trsm` and a `gemm` per panel, so the bulk of the
/// work runs at `gemm` speed. A pivot below `singular_tol` marks the result
/// singular; the factorization still completes so the caller can inspect it.
inline lu_result lu(const mat &A) {
    constexpr real singular_tol = 1e-14;
    const idx n = A.rows();
    lu_result f;
    f.LU = A;
    f.piv.resize(n);
    f.singular = !kernel::lu_factor_blocked(f.LU.data(), f.piv.data(), n);
    for (idx k = 0; k < n && !f.singular; ++k) {
        f.singular = std::abs(f.LU(k, k)) < singular_tol;
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
#if defined(NUMERICS_LAPACK_DEFAULT)
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
#if defined(NUMERICS_LAPACK_DEFAULT)
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
    // P B, then L Y = P B and U X = Y, both as blocked triangular solves.
    const idx nrhs = B.cols();
    for (idx k = 0; k < n; ++k) {
        if (f.piv[k] != k) {
            kernel::swap_rows(X.data(), nrhs, k, f.piv[k], nrhs);
        }
    }
    kernel::trsm_unit_lower_inplace(X.data(), nrhs, f.LU.data(), n, n, nrhs);
    kernel::trsm_upper_inplace(X.data(), nrhs, f.LU.data(), n, n, nrhs);
#endif
}

/// Solve \f$A^T X = B\f$ for several right-hand sides from \f$PA = LU\f$:
/// \f$A^T = U^T L^T P\f$, so \f$U^T Q = B\f$, \f$L^T Y = Q\f$, \f$X = P^T Y\f$.
inline void lu_solve_transpose(const lu_result &f, const mat &B, mat &X) {
    const idx n = f.LU.rows();
    if (f.LU.cols() != n || B.rows() != n) {
        throw std::invalid_argument("lu_solve_transpose: dimension mismatch");
    }
    X = B;
    const idx nrhs = B.cols();
#if defined(NUMERICS_LAPACK_DEFAULT)
    array<lapack_int> pivots(n);
    for (idx index = 0; index < n; ++index) {
        pivots[index] = static_cast<lapack_int>(f.piv[index] + 1);
    }
    const int info =
        LAPACKE_dgetrs(LAPACK_ROW_MAJOR, 'T', static_cast<lapack_int>(n),
                       static_cast<lapack_int>(nrhs), f.LU.data(), static_cast<lapack_int>(n),
                       pivots.data(), X.data(), static_cast<lapack_int>(nrhs));
    if (info != 0) {
        throw std::runtime_error("lu_solve_transpose: LAPACK block solve failed");
    }
#else
    kernel::trsm_upper_transpose_inplace(X.data(), nrhs, f.LU.data(), n, n, nrhs);
    kernel::trsm_unit_lower_transpose_inplace(X.data(), nrhs, f.LU.data(), n, n, nrhs);
    // X = P^T Y: undo the row interchanges in reverse order.
    for (idx step = n; step-- > 0;) {
        if (f.piv[step] != step) {
            kernel::swap_rows(X.data(), nrhs, step, f.piv[step], nrhs);
        }
    }
#endif
}

inline void lu_solve_transpose(const lu_result &f, const vec &b, vec &x) {
    const idx n = f.LU.rows();
    if (f.LU.cols() != n || b.size() != n) {
        throw std::invalid_argument("lu_solve_transpose: dimension mismatch");
    }
    mat column(n, 1, 0.0);
    for (idx row = 0; row < n; ++row) {
        column(row, 0) = b[row];
    }
    mat solution;
    lu_solve_transpose(f, column, solution);
    x = vec(n, 0.0);
    for (idx row = 0; row < n; ++row) {
        x[row] = solution(row, 0);
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
#if defined(NUMERICS_LAPACK_DEFAULT)
    array<lapack_int> ipiv(n);
    for (idx i = 0; i < n; ++i)
        ipiv[i] = static_cast<lapack_int>(f.piv[i] + 1);
    LAPACKE_dgetri(LAPACK_ROW_MAJOR, static_cast<lapack_int>(n), inv.data(),
                   static_cast<lapack_int>(n), ipiv.data());
#else
    array<real> work(n);
    kernel::lu_invert(inv.data(), f.LU.data(), f.piv.data(), n, work.data());
#endif
    return inv;
}

} // namespace num
