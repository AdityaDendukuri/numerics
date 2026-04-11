/// @file core/backends/blas/matrix.cpp
/// @brief BLAS backend  -- cblas level-2/3 matrix operations
///
/// Delegates to the system BLAS (OpenBLAS, MKL, Apple Accelerate, ...).
/// When NUMERICS_HAS_BLAS is not defined, falls back to the blocked backend
/// and emits a one-time stderr warning.

#include "core/matrix.hpp"
#include "../seq/impl.hpp"
#include <cstdio>

#ifdef NUMERICS_HAS_BLAS
    #include <cblas.h>
#endif

namespace {
void warn_blas_unavailable() {
#ifndef NUMERICS_HAS_BLAS
    static bool warned = false;
    if (!warned) {
        warned = true;
        std::fprintf(
            stderr,
            "[numerics] WARNING: Backend::blas requested but BLAS was not "
            "found at "
            "configure time.\n"
            "           Falling back to Backend::blocked (cache-blocked).\n"
            "           Install OpenBLAS and reconfigure: "
            "apt install libopenblas-dev  |  brew install openblas\n");
    }
#endif
}
} // namespace

namespace num::backends::blas {

void matmul(const Matrix& A, const Matrix& B, Matrix& C) {
    warn_blas_unavailable();
#ifdef NUMERICS_HAS_BLAS
    cblas_dgemm(CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                static_cast<int>(A.rows()),
                static_cast<int>(B.cols()),
                static_cast<int>(A.cols()),
                1.0,
                A.data(),
                static_cast<int>(A.cols()),
                B.data(),
                static_cast<int>(B.cols()),
                0.0,
                C.data(),
                static_cast<int>(C.cols()));
#else
    num::backends::seq::matmul_blocked(A, B, C, 64);
#endif
}

void matvec(const Matrix& A, const Vector& x, Vector& y) {
    warn_blas_unavailable();
#ifdef NUMERICS_HAS_BLAS
    cblas_dgemv(CblasRowMajor,
                CblasNoTrans,
                static_cast<int>(A.rows()),
                static_cast<int>(A.cols()),
                1.0,
                A.data(),
                static_cast<int>(A.cols()),
                x.data(),
                1,
                0.0,
                y.data(),
                1);
#else
    num::backends::seq::matvec(A, x, y);
#endif
}

void matadd(real          alpha,
            const Matrix& A,
            real          beta,
            const Matrix& B,
            Matrix&       C) {
    warn_blas_unavailable();
#ifdef NUMERICS_HAS_BLAS
    cblas_dcopy(static_cast<int>(A.size()), A.data(), 1, C.data(), 1);
    cblas_dscal(static_cast<int>(C.size()), alpha, C.data(), 1);
    cblas_daxpy(static_cast<int>(B.size()), beta, B.data(), 1, C.data(), 1);
#else
    num::backends::seq::matadd(alpha, A, beta, B, C);
#endif
}

} // namespace num::backends::blas
