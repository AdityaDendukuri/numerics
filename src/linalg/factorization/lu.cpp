/// @file linalg/factorization/lu.cpp
/// @brief LU factorization dispatcher + implementations (sequential & LAPACK).

#include "linalg/factorization/lu.hpp"
#include "linalg/matrix_utils.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(NUMERICS_HAS_LAPACK)
#include "core/parallel/lapack_wrapper.hpp"
#endif

namespace num {

namespace backends {

namespace seq {
LUResult lu(const Matrix &A) {
    constexpr real singular_tol = 1e-14;
    const idx n = A.rows();
    LUResult f;
    f.LU = A;
    f.piv.resize(n);
    f.singular = false;

    Matrix &M = f.LU;

    for (idx k = 0; k < n; ++k) {
        idx pivot_row = k;
        real pivot_val = std::abs(M(k, k));
        for (idx i = k + 1; i < n; ++i) {
            real v = std::abs(M(i, k));
            if (v > pivot_val) {
                pivot_val = v;
                pivot_row = i;
            }
        }
        f.piv[k] = pivot_row;

        if (pivot_row != k) {
            for (idx j = 0; j < n; ++j) {
                std::swap(M(k, j), M(pivot_row, j));
            }
        }

        if (std::abs(M(k, k)) < singular_tol) {
            f.singular = true;
            continue;
        }

        const real inv_ukk = real(1) / M(k, k);
        for (idx i = k + 1; i < n; ++i) {
            M(i, k) *= inv_ukk;
        }

        for (idx i = k + 1; i < n; ++i) {
            const real lik = M(i, k);
            for (idx j = k + 1; j < n; ++j) {
                M(i, j) -= lik * M(k, j);
            }
        }
    }

    return f;
}
} // namespace seq

namespace lapack {
LUResult lu(const Matrix &A) {
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

LUResult lu(const Matrix &A, Backend backend) {
    switch (backend) {
    case Backend::lapack:
        return backends::lapack::lu(A);
    default:
        return backends::seq::lu(A);
    }
}

void lu_solve(const LUResult &f, const Vector &b, Vector &x) {
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

void lu_solve(const LUResult &f, const Matrix &B, Matrix &X) {
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

void lu_solve_transpose(const LUResult &f, const Vector &b, Vector &x) {
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

void lu_solve_transpose(const LUResult &f, const Matrix &B, Matrix &X) {
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

void solve_in_place(const LUResult &f, Vector &right_hand_side) {
    Vector result(right_hand_side.size(), 0.0);
    lu_solve(f, right_hand_side, result);
    right_hand_side = std::move(result);
}

void solve_in_place(const LUResult &f, Matrix &right_hand_sides) {
    Matrix result;
    lu_solve(f, right_hand_sides, result);
    right_hand_sides = std::move(result);
}

real lu_det(const LUResult &f) {
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

Matrix lu_inv(const LUResult &f) {
    const idx n = f.LU.rows();
    const Matrix identity_matrix = identity(n);
    Matrix inv;
    lu_solve(f, identity_matrix, inv);
    return inv;
}

} // namespace num
