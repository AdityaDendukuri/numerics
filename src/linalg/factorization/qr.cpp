/// @file linalg/factorization/qr.cpp
/// @brief QR dispatcher + qr_solve.
///
/// Backend routing:
///   Backend::lapack  -> backends::lapack::qr  (dgeqrf + dorgqr, blocked BLAS-3)
///   everything else  -> backends::seq::qr     (Householder reflections)

#include "linalg/factorization/qr.hpp"
#include "backends/seq/impl.hpp"
#include "backends/lapack/impl.hpp"

namespace num {

QRResult qr(const Matrix& A, Backend backend) {
    switch (backend) {
    case Backend::lapack:
        return backends::lapack::qr(A);
    default:
        return backends::seq::qr(A);
    }
}

// qr_solve()   --  least-squares via Q^T multiply + back-substitution

void qr_solve(const QRResult& f, const Vector& b, Vector& x) {
    const idx m = f.Q.rows();
    const idx n = f.R.cols();

    Vector y(m, real(0));
    for (idx i = 0; i < m; ++i)
        for (idx j = 0; j < m; ++j)
            y[i] += f.Q(j, i) * b[j];

    Vector xv(n, real(0));
    for (idx i = n; i-- > 0; ) {
        xv[i] = y[i];
        for (idx j = i + 1; j < n; ++j)
            xv[i] -= f.R(i, j) * xv[j];
        xv[i] /= f.R(i, i);
    }

    x = std::move(xv);
}

} // namespace num
