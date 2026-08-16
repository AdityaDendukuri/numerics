/// @file linalg/factorization/lu.cpp
/// @brief LU factorization dispatcher + implementations (sequential & LAPACK).

#include "linalg/factorization/lu.hpp"
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
LUResult lu(const Matrix& A) {
    constexpr real singular_tol = 1e-14;
    const idx n = A.rows();
    LUResult f;
    f.LU = A;
    f.piv.resize(n);
    f.singular = false;

    Matrix& M = f.LU;

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

        if (pivot_row != k)
            for (idx j = 0; j < n; ++j)
                std::swap(M(k, j), M(pivot_row, j));

        if (std::abs(M(k, k)) < singular_tol) {
            f.singular = true;
            continue;
        }

        const real inv_ukk = real(1) / M(k, k);
        for (idx i = k + 1; i < n; ++i)
            M(i, k) *= inv_ukk;

        for (idx i = k + 1; i < n; ++i) {
            const real lik = M(i, k);
            for (idx j = k + 1; j < n; ++j)
                M(i, j) -= lik * M(k, j);
        }
    }

    return f;
}
} // namespace seq

namespace lapack {
LUResult lu(const Matrix& A) {
#if defined(NUMERICS_HAS_LAPACK)
    const idx n = A.rows();
    LUResult f;
    f.LU = A;
    f.piv.resize(n);
    f.singular = false;

    std::vector<lapack_int> ipiv(n);
    int info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR,
                              static_cast<lapack_int>(n),
                              static_cast<lapack_int>(n),
                              f.LU.data(),
                              static_cast<lapack_int>(n),
                              ipiv.data());
    if (info < 0)
        throw std::runtime_error("lu (lapack): dgetrf argument error, info="
                                 + std::to_string(info));
    if (info > 0)
        f.singular = true;

    for (idx k = 0; k < n; ++k)
        f.piv[k] = static_cast<idx>(ipiv[k] - 1);

    return f;
#else
    return seq::lu(A);
#endif
}
} // namespace lapack

} // namespace backends

LUResult lu(const Matrix& A, Backend backend) {
    switch (backend) {
        case Backend::lapack:
            return backends::lapack::lu(A);
        default:
            return backends::seq::lu(A);
    }
}

void lu_solve(const LUResult& f, const Vector& b, Vector& x) {
    const idx n = f.LU.rows();
    const Matrix& M = f.LU;
    Vector y = b;

    for (idx k = 0; k < n; ++k)
        if (f.piv[k] != k)
            std::swap(y[k], y[f.piv[k]]);

    for (idx i = 1; i < n; ++i)
        for (idx j = 0; j < i; ++j)
            y[i] -= M(i, j) * y[j];

    for (idx i = n; i-- > 0;) {
        for (idx j = i + 1; j < n; ++j)
            y[i] -= M(i, j) * y[j];
        y[i] /= M(i, i);
    }

    x = std::move(y);
}

void lu_solve(const LUResult& f, const Matrix& B, Matrix& X) {
    const idx nrhs = B.cols();
    const idx n = B.rows();
    Vector col(n), xcol(n);
    for (idx j = 0; j < nrhs; ++j) {
        for (idx i = 0; i < n; ++i)
            col[i] = B(i, j);
        lu_solve(f, col, xcol);
        for (idx i = 0; i < n; ++i)
            X(i, j) = xcol[i];
    }
}

real lu_det(const LUResult& f) {
    const idx n = f.LU.rows();
    real det = real(1);
    for (idx i = 0; i < n; ++i)
        det *= f.LU(i, i);
    idx swaps = 0;
    for (idx k = 0; k < n; ++k)
        if (f.piv[k] != k)
            ++swaps;
    return (swaps % 2 == 0) ? det : -det;
}

Matrix lu_inv(const LUResult& f) {
    const idx n = f.LU.rows();
    Matrix inv(n, n, real(0));
    Vector e(n, real(0)), col(n);
    for (idx j = 0; j < n; ++j) {
        e[j] = real(1);
        lu_solve(f, e, col);
        for (idx i = 0; i < n; ++i)
            inv(i, j) = col[i];
        e[j] = real(0);
    }
    return inv;
}

} // namespace num
