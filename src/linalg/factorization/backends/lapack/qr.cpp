/// @file linalg/factorization/backends/lapack/qr.cpp
/// @brief LAPACK QR factorization via LAPACKE_dgeqrf + LAPACKE_dorgqr.
#include "impl.hpp"
#include "../seq/impl.hpp"
#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(NUMERICS_HAS_LAPACK)
    #include <lapacke.h>
#endif

namespace num::backends::lapack {

QRResult qr(const Matrix& A) {
#if defined(NUMERICS_HAS_LAPACK)
    const idx m = A.rows(), n = A.cols();
    const idx k = std::min(m, n);

    Matrix              R = A;
    std::vector<double> tau(k);

    int info =
        LAPACKE_dgeqrf(LAPACK_ROW_MAJOR,
                       static_cast<lapack_int>(m),
                       static_cast<lapack_int>(n),
                       R.data(),
                       static_cast<lapack_int>(n), // lda = cols (row-major)
                       tau.data());
    if (info != 0)
        throw std::runtime_error("qr (lapack): dgeqrf failed, info="
                                 + std::to_string(info));

    // Extract upper triangle as R
    Matrix Rmat = R;
    for (idx i = 1; i < m; ++i)
        for (idx j = 0; j < std::min(i, n); ++j)
            Rmat(i, j) = 0.0;

    // Build full Q from stored reflectors via dorgqr
    Matrix Q(m, m, 0.0);
    for (idx j = 0; j < k; ++j)
        for (idx i = 0; i < m; ++i)
            Q(i, j) = R(i, j);

    info = LAPACKE_dorgqr(LAPACK_ROW_MAJOR,
                          static_cast<lapack_int>(m),
                          static_cast<lapack_int>(m),
                          static_cast<lapack_int>(k),
                          Q.data(),
                          static_cast<lapack_int>(m), // lda = cols = m (square)
                          tau.data());
    if (info != 0)
        throw std::runtime_error("qr (lapack): dorgqr failed, info="
                                 + std::to_string(info));

    return {std::move(Q), std::move(Rmat)};
#else
    return seq::qr(A);
#endif
}

} // namespace num::backends::lapack
