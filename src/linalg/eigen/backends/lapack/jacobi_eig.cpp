/// @file linalg/eigen/backends/lapack/jacobi_eig.cpp
/// @brief LAPACK symmetric eigensolver via LAPACKE_dsyevd (divide-and-conquer).
#include "impl.hpp"
#include "../seq/impl.hpp"
#include <stdexcept>
#include <string>

#if defined(NUMERICS_HAS_LAPACK)
    #include <lapacke.h>
#endif

namespace num::backends::lapack {

EigenResult eig_sym(const Matrix& A) {
#if defined(NUMERICS_HAS_LAPACK)
    if (A.rows() != A.cols())
        throw std::invalid_argument("eig_sym: matrix must be square");
    idx    n  = A.rows();
    Matrix Aw = A; // dsyevd overwrites A with eigenvectors
    Vector w(n);
    int    info = LAPACKE_dsyevd(LAPACK_ROW_MAJOR,
                              'V',
                              'U',
                              static_cast<lapack_int>(n),
                              Aw.data(),
                              static_cast<lapack_int>(n),
                              w.data());
    if (info != 0)
        throw std::runtime_error("eig_sym (lapack): dsyevd failed, info="
                                 + std::to_string(info));
    // dsyevd returns eigenvalues ascending; eigenvectors in columns of Aw
    return {w, Aw, 0, true};
#else
    return seq::eig_sym(A, 1e-12, 100);
#endif
}

} // namespace num::backends::lapack
