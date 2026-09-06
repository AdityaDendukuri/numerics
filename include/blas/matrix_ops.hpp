/// @file blas/matrix_ops.hpp
/// @brief BLAS-accelerated Level-2/3 dense matrix operations.
#pragma once

#include "blas/vector_ops.hpp"
#include "container/matrix.hpp"
#include "container/vector.hpp"
#include "core/types.hpp"
#include "kernel/kernel.hpp"

#if defined(NUMERICS_HAS_BLAS)
#include <cblas.h>
#endif

namespace num::blas {

inline void matmul(const mat &A, const mat &B, mat &C) {
#if defined(NUMERICS_HAS_BLAS)
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<int>(A.rows()),
                static_cast<int>(B.cols()), static_cast<int>(A.cols()), 1.0, A.data(),
                static_cast<int>(A.cols()), B.data(), static_cast<int>(B.cols()), 0.0, C.data(),
                static_cast<int>(C.cols()));
#else
    warn_unavailable();
    kernel::gemm(C.data(), A.data(), B.data(), real(1), real(0), A.rows(), B.cols(), A.cols());
#endif
}

inline void matvec(const mat &A, const vec &x, vec &y) {
#if defined(NUMERICS_HAS_BLAS)
    cblas_dgemv(CblasRowMajor, CblasNoTrans, static_cast<int>(A.rows()), static_cast<int>(A.cols()),
                1.0, A.data(), static_cast<int>(A.cols()), x.data(), 1, 0.0, y.data(), 1);
#else
    warn_unavailable();
    kernel::matvec(y.data(), A.data(), x.data(), A.rows(), A.cols());
#endif
}

inline void matadd(real alpha, const mat &A, real beta, const mat &B, mat &C) {
#if defined(NUMERICS_HAS_BLAS)
    cblas_dcopy(static_cast<int>(A.size()), A.data(), 1, C.data(), 1);
    cblas_dscal(static_cast<int>(C.size()), alpha, C.data(), 1);
    cblas_daxpy(static_cast<int>(B.size()), beta, B.data(), 1, C.data(), 1);
#else
    warn_unavailable();
    kernel::axpbyz(C.data(), A.data(), B.data(), alpha, beta, A.size());
#endif
}

} // namespace num::blas
