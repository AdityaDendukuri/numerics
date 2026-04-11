/// @file linalg/svd/backends/lapack/svd.cpp
/// @brief LAPACK SVD via LAPACKE_dgesdd (divide-and-conquer).
#include "impl.hpp"
#include "../seq/impl.hpp"
#include <algorithm>
#include <stdexcept>
#include <string>

#if defined(NUMERICS_HAS_LAPACK)
    #include <lapacke.h>
#endif

namespace num::backends::lapack {

SVDResult svd(const Matrix& A_in) {
#if defined(NUMERICS_HAS_LAPACK)
    const idx m = A_in.rows(), n = A_in.cols();
    const idx r  = std::min(m, n);
    Matrix    Aw = A_in; // dgesdd overwrites A
    Vector    S(r);
    // Economy mode ('S'): U is m x r (lda=m), Vt is r x n (lda=r).
    // Matrix lda equals its row count, so declare Vt as (r x n) not (n x n).
    Matrix U(m, r);
    Matrix Vt(r, n);

    int info =
        LAPACKE_dgesdd(LAPACK_ROW_MAJOR,
                       'S',
                       static_cast<lapack_int>(m),
                       static_cast<lapack_int>(n),
                       Aw.data(),
                       static_cast<lapack_int>(n), // lda = cols (row-major)
                       S.data(),
                       U.data(),
                       static_cast<lapack_int>(r),  // ldu = r (cols of U)
                       Vt.data(),
                       static_cast<lapack_int>(n)); // ldvt = n (cols of Vt)
    if (info != 0)
        throw std::runtime_error("svd (lapack): dgesdd failed, info="
                                 + std::to_string(info));

    return {std::move(U), std::move(S), std::move(Vt), 0, true};
#else
    return seq::svd(A_in, 1e-12, 100);
#endif
}

} // namespace num::backends::lapack
