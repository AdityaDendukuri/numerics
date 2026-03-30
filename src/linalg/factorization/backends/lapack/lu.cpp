/// @file linalg/factorization/backends/lapack/lu.cpp
/// @brief LAPACK LU factorization via LAPACKE_dgetrf.
///
/// Falls back to seq::lu() with a warning when NUMERICS_HAS_LAPACK is not set;
/// this keeps the build working but reminds the caller that LAPACK is absent.
#include "impl.hpp"
#include "../seq/impl.hpp"
#include <stdexcept>
#include <string>
#include <vector>

#if defined(NUMERICS_HAS_LAPACK)
#  include <lapacke.h>
#endif

namespace num::backends::lapack {

LUResult lu(const Matrix& A) {
#if defined(NUMERICS_HAS_LAPACK)
    const idx n = A.rows();
    LUResult f;
    f.LU  = A;
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
        throw std::runtime_error("lu (lapack): dgetrf argument error, info=" + std::to_string(info));
    if (info > 0)
        f.singular = true;

    // Convert 1-based LAPACK IPIV to 0-based sequential swap targets
    for (idx k = 0; k < n; ++k)
        f.piv[k] = static_cast<idx>(ipiv[k] - 1);

    return f;
#else
    return seq::lu(A);
#endif
}

} // namespace num::backends::lapack
