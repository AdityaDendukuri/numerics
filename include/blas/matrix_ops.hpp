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
#ifndef NUMERICS_CBLAS_HEADER
#define NUMERICS_CBLAS_HEADER <cblas.h>
#endif
#include NUMERICS_CBLAS_HEADER
#endif

namespace num::blas {

inline void matmul(const mat<real> &A, const mat<real> &B, mat<real> &C) {
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
                                 transB ? idx{1} : ldb, transB ? ldb : idx{1}, alpha, beta, m, n, k,
                                 scratch.get());
#endif
}

/// @brief `C <- alpha op(A) op(B) + beta C` for matrices; C must already be `m x n`.
inline void gemm(real alpha, const mat<real> &A, bool transA, const mat<real> &B, bool transB,
                 real beta, mat<real> &C) {
    const idx m = transA ? A.cols() : A.rows();
    const idx k = transA ? A.rows() : A.cols();
    const idx n = transB ? B.rows() : B.cols();
    if ((transB ? B.cols() : B.rows()) != k || C.rows() != m || C.cols() != n) {
        throw std::invalid_argument("gemm: dimensions do not agree");
    }
    gemm(alpha, A.data(), A.cols(), transA, B.data(), B.cols(), transB, beta, C.data(), C.cols(), m,
         n, k);
}

/// @brief Solve `op(T) X = B` in place for a triangular `T` and row-major `X`.
///
/// This is the explicit external-BLAS alternative to the in-tree triangular kernels.
inline void trsm_left(const real *T, idx ldt, real *X, idx ldx, idx n, idx nrhs, bool upper,
                      bool transpose, bool unit_diagonal) {
#if defined(NUMERICS_HAS_BLAS)
    cblas_dtrsm(CblasRowMajor, CblasLeft, upper ? CblasUpper : CblasLower,
                transpose ? CblasTrans : CblasNoTrans, unit_diagonal ? CblasUnit : CblasNonUnit,
                static_cast<int>(n), static_cast<int>(nrhs), 1.0, T, static_cast<int>(ldt), X,
                static_cast<int>(ldx));
#else
    warn_unavailable();
    if (upper) {
        if (transpose)
            kernel::trsm_upper_transpose_inplace(X, ldx, T, ldt, n, nrhs, unit_diagonal);
        else
            kernel::trsm_upper_inplace(X, ldx, T, ldt, n, nrhs, unit_diagonal);
    } else if (transpose) {
        if (unit_diagonal)
            kernel::trsm_unit_lower_transpose_inplace(X, ldx, T, ldt, n, nrhs);
        else
            kernel::trsm_lower_transpose_inplace(X, ldx, T, n, nrhs);
    } else if (unit_diagonal) {
        kernel::trsm_unit_lower_inplace(X, ldx, T, ldt, n, nrhs);
    } else {
        kernel::trsm_lower_inplace(X, ldx, T, n, nrhs);
    }
#endif
}

/// @brief Solve `X L^T = B` in place for a row-major panel and lower-triangular `L`.
inline void trsm_right_lower_transpose(const real *L, idx ldl, real *X, idx ldx, idx rows,
                                       idx n) {
#if defined(NUMERICS_HAS_BLAS)
    cblas_dtrsm(CblasRowMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit,
                static_cast<int>(rows), static_cast<int>(n), 1.0, L, static_cast<int>(ldl), X,
                static_cast<int>(ldx));
#else
    warn_unavailable();
    kernel::trsm_lower_transpose_right_inplace(X, ldx, L, ldl, rows, n);
#endif
}

/// @brief Update the lower triangle, `C <- alpha A A^T + beta C`.
inline void syrk_lower(real alpha, const real *A, idx lda, real beta, real *C, idx ldc, idx rows,
                       idx columns) {
#if defined(NUMERICS_HAS_BLAS)
    cblas_dsyrk(CblasRowMajor, CblasLower, CblasNoTrans, static_cast<int>(rows),
                static_cast<int>(columns), alpha, A, static_cast<int>(lda), beta, C,
                static_cast<int>(ldc));
#else
    warn_unavailable();
    kernel::syrk_lower(C, ldc, A, lda, alpha, beta, rows, columns);
#endif
}

inline void matvec(const mat<real> &A, const vec<real> &x, vec<real> &y) {
#if defined(NUMERICS_HAS_BLAS)
    cblas_dgemv(CblasRowMajor, CblasNoTrans, static_cast<int>(A.rows()), static_cast<int>(A.cols()),
                1.0, A.data(), static_cast<int>(A.cols()), x.data(), 1, 0.0, y.data(), 1);
#else
    warn_unavailable();
    kernel::matvec(y.data(), A.data(), x.data(), A.rows(), A.cols());
#endif
}

inline void matadd(real alpha, const mat<real> &A, real beta, const mat<real> &B, mat<real> &C) {
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
