/// @file eigen/eig.cpp
/// @brief Full symmetric eigendecomposition dispatcher.
///
/// Backend routing:
///   Backend::lapack  -> backends::lapack::eig_sym  (LAPACKE_dsyevd,
///   divide-and-conquer) Backend::omp     -> backends::omp::eig_sym     (cyclic
///   Jacobi, parallel inner loops) everything else  -> backends::seq::eig_sym
///   (cyclic Jacobi, sequential)

#include "linalg/eigen/jacobi_eig.hpp"
#include "backends/seq/impl.hpp"
#include "backends/omp/impl.hpp"
#include "backends/lapack/impl.hpp"

namespace num {

EigenResult eig_sym(const Matrix& A,
                    real          tol,
                    idx           max_sweeps,
                    Backend       backend) {
    switch (backend) {
        case Backend::lapack:
            return backends::lapack::eig_sym(A);
        case Backend::omp:
            return backends::omp::eig_sym(A, tol, max_sweeps);
        default:
            return backends::seq::eig_sym(A, tol, max_sweeps);
    }
}

} // namespace num
