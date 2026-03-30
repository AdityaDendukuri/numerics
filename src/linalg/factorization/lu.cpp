/// @file linalg/factorization/lu.cpp
/// @brief LU dispatcher + utility functions (lu_solve, lu_det, lu_inv).
///
/// Backend routing:
///   Backend::lapack  -> backends::lapack::lu  (LAPACKE_dgetrf, blocked BLAS-3)
///   everything else  -> backends::seq::lu     (Doolittle, partial pivoting)
///
/// Adding an omp backend: create backends/omp/lu.cpp, include its impl.hpp here,
/// and add the case Backend::omp below.

#include "linalg/factorization/lu.hpp"
#include "backends/seq/impl.hpp"
#include "backends/lapack/impl.hpp"

namespace num {

LUResult lu(const Matrix& A, Backend backend) {
    switch (backend) {
    case Backend::lapack:
        return backends::lapack::lu(A);
    default:
        return backends::seq::lu(A);
    }
}

// lu_solve()   --  apply P, then forward/backward substitution

void lu_solve(const LUResult& f, const Vector& b, Vector& x) {
    const idx n    = f.LU.rows();
    const Matrix& M = f.LU;

    Vector y = b;

    for (idx k = 0; k < n; ++k)
        if (f.piv[k] != k)
            std::swap(y[k], y[f.piv[k]]);

    for (idx i = 1; i < n; ++i)
        for (idx j = 0; j < i; ++j)
            y[i] -= M(i, j) * y[j];

    for (idx i = n; i-- > 0; ) {
        for (idx j = i + 1; j < n; ++j)
            y[i] -= M(i, j) * y[j];
        y[i] /= M(i, i);
    }

    x = std::move(y);
}

void lu_solve(const LUResult& f, const Matrix& B, Matrix& X) {
    const idx nrhs = B.cols();
    const idx n    = B.rows();
    Vector col(n), xcol(n);
    for (idx j = 0; j < nrhs; ++j) {
        for (idx i = 0; i < n; ++i) col[i] = B(i, j);
        lu_solve(f, col, xcol);
        for (idx i = 0; i < n; ++i) X(i, j) = xcol[i];
    }
}

// lu_det()   --  determinant from diagonal of U and pivot parity

real lu_det(const LUResult& f) {
    const idx n = f.LU.rows();
    real det = real(1);
    for (idx i = 0; i < n; ++i)
        det *= f.LU(i, i);
    idx swaps = 0;
    for (idx k = 0; k < n; ++k)
        if (f.piv[k] != k) ++swaps;
    return (swaps % 2 == 0) ? det : -det;
}

// lu_inv()   --  A^{-1} by solving A * X = I column by column

Matrix lu_inv(const LUResult& f) {
    const idx n = f.LU.rows();
    Matrix inv(n, n, real(0));
    Vector e(n, real(0)), col(n);
    for (idx j = 0; j < n; ++j) {
        e[j] = real(1);
        lu_solve(f, e, col);
        for (idx i = 0; i < n; ++i) inv(i, j) = col[i];
        e[j] = real(0);
    }
    return inv;
}

} // namespace num
