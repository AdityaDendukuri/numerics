/// @file blas/matrix_ops.hpp
/// @brief Dense matrix operations through an external BLAS.
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

/// @brief Matrix-matrix product (GEMM): \f$C \leftarrow \alpha\, op(A)\, op(B) + \beta C\f$.
/// Row-major with leading dimensions; \f$op\f$ transposes when its flag is set.
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

/// @brief Matrix-matrix product (GEMM): \f$C \leftarrow \alpha\, op(A)\, op(B) + \beta C\f$.
/// `C` must already be sized.
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

/// @brief Triangular solve, left side (TRSM): \f$X \leftarrow op(T)^{-1} X\f$.
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

/// @brief Triangular solve, right side (TRSM): \f$X \leftarrow X L^{-T}\f$.
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

/// @brief Symmetric rank-k update (SYRK), lower: \f$C \leftarrow \alpha A A^T + \beta C\f$.
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

/// @brief Rank-1 update (GER): \f$A \leftarrow A + \alpha x y^T\f$.
inline void ger(real alpha, const vec<real> &x, const vec<real> &y, mat<real> &A) {
    if (A.rows() != x.size() || A.cols() != y.size()) {
        throw std::invalid_argument("ger: dimensions do not agree");
    }
#if defined(NUMERICS_HAS_BLAS)
    cblas_dger(CblasRowMajor, static_cast<int>(x.size()), static_cast<int>(y.size()), alpha,
               x.data(), 1, y.data(), 1, A.data(), static_cast<int>(A.cols()));
#else
    warn_unavailable();
    kernel::ger(A.data(), x.data(), y.data(), alpha, x.size(), y.size());
#endif
}

namespace detail {
inline idx trsv_prepare(const mat<real> &T, const vec<real> &b, vec<real> &x) {
    const idx n = T.rows();
    if (T.cols() != n || b.size() != n) {
        throw std::invalid_argument("trsv: dimension mismatch");
    }
    if (&x != &b) {
        x = b;
    }
    return n;
}
} // namespace detail

/// @brief Lower triangular solve (TRSV): \f$x \leftarrow L^{-1} b\f$. `x` may alias `b`.
inline void trsv_lower(const mat<real> &L, const vec<real> &b, vec<real> &x) {
    const idx n = detail::trsv_prepare(L, b, x);
#if defined(NUMERICS_HAS_BLAS)
    cblas_dtrsv(CblasRowMajor, CblasLower, CblasNoTrans, CblasNonUnit, static_cast<int>(n),
                L.data(), static_cast<int>(n), x.data(), 1);
#else
    warn_unavailable();
    kernel::trsv_lower_inplace(x.data(), L.data(), n);
#endif
}

/// @brief Upper triangular solve (TRSV): \f$x \leftarrow U^{-1} b\f$. `x` may alias `b`.
inline void trsv_upper(const mat<real> &U, const vec<real> &b, vec<real> &x) {
    const idx n = detail::trsv_prepare(U, b, x);
#if defined(NUMERICS_HAS_BLAS)
    cblas_dtrsv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, static_cast<int>(n),
                U.data(), static_cast<int>(n), x.data(), 1);
#else
    warn_unavailable();
    kernel::trsv_upper_inplace(x.data(), U.data(), n);
#endif
}

} // namespace num::blas
