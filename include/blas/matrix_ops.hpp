/// @file blas/matrix_ops.hpp
/// @brief BLAS-accelerated Level-2/3 dense matrix operations.
#pragma once

#include "blas/vector_ops.hpp"
#include "container/matrix.hpp"
#include "container/vector.hpp"
#include "core/types.hpp"
#include "kernel/kernel.hpp"
#include <stdexcept>

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

/// @brief `C <- alpha op(A) op(B) + beta C` on row-major storage with leading
/// dimensions, where `op` transposes when the flag is set: `op(A)` is `m x k`,
/// `op(B)` is `k x n`, C is `m x n`. No operand is copied.
inline void gemm(real alpha, const real *A, idx lda, bool transA, const real *B, idx ldb,
                 bool transB, real beta, real *C, idx ldc, idx m, idx n, idx k) {
#if defined(NUMERICS_HAS_BLAS)
    cblas_dgemm(CblasRowMajor, transA ? CblasTrans : CblasNoTrans,
                transB ? CblasTrans : CblasNoTrans, static_cast<int>(m), static_cast<int>(n),
                static_cast<int>(k), alpha, A, static_cast<int>(lda), B, static_cast<int>(ldb),
                beta, C, static_cast<int>(ldc));
#else
    warn_unavailable();
    const kernel::detail::gemm_scratch<real> scratch;
    kernel::detail::gemm_strided(C, ldc, A, transA ? idx{1} : lda, transA ? lda : idx{1}, B,
                                 transB ? idx{1} : ldb, transB ? ldb : idx{1}, alpha, beta, m, n,
                                 k, scratch.get());
#endif
}

/// @brief `C <- alpha op(A) op(B) + beta C` for matrices; C must already be `m x n`.
inline void gemm(real alpha, const mat &A, bool transA, const mat &B, bool transB, real beta,
                 mat &C) {
    const idx m = transA ? A.cols() : A.rows();
    const idx k = transA ? A.rows() : A.cols();
    const idx n = transB ? B.rows() : B.cols();
    if ((transB ? B.cols() : B.rows()) != k || C.rows() != m || C.cols() != n) {
        throw std::invalid_argument("gemm: dimensions do not agree");
    }
    gemm(alpha, A.data(), A.cols(), transA, B.data(), B.cols(), transB, beta, C.data(), C.cols(),
         m, n, k);
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
