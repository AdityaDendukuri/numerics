/// @file lapack_wrapper.hpp
/// @brief Standard C/C++ LAPACKE header for cross-platform Linux and macOS build targets.
#pragma once

#include "core/types.hpp"
#include "core/policy.hpp"
#include <algorithm>
#include <vector>

#if defined(NUMERICS_HAS_LAPACK)

#if defined(NUMERICS_LAPACK_ACCELERATE)
#include <Accelerate/Accelerate.h>

using lapack_int = int;
#define LAPACK_ROW_MAJOR 101
#define LAPACK_COL_MAJOR 102

inline int LAPACKE_dgetrf(int matrix_layout, lapack_int m, lapack_int n, double *a, lapack_int lda,
                          lapack_int *ipiv) {
    int info = 0;
    dgetrf_(&n, &m, a, &lda, ipiv, &info);
    return info;
}

inline int LAPACKE_dgeqrf(int matrix_layout, lapack_int m, lapack_int n, double *a, lapack_int lda,
                          double *tau) {
    int info = 0;
    int lwork = -1;
    double work_query = 0.0;
    dgeqrf_(&n, &m, a, &lda, tau, &work_query, &lwork, &info);
    lwork = static_cast<int>(work_query);
    array<double> work(std::max(1, lwork));
    dgeqrf_(&n, &m, a, &lda, tau, work.data(), &lwork, &info);
    return info;
}

inline int LAPACKE_dorgqr(int matrix_layout, lapack_int m, lapack_int n, lapack_int k, double *a,
                          lapack_int lda, const double *tau) {
    int info = 0;
    int lwork = -1;
    double work_query = 0.0;
    dorgqr_(&n, &m, &k, a, &lda, tau, &work_query, &lwork, &info);
    lwork = static_cast<int>(work_query);
    array<double> work(std::max(1, lwork));
    dorgqr_(&n, &m, &k, a, &lda, tau, work.data(), &lwork, &info);
    return info;
}

inline int LAPACKE_dgehrd(int matrix_layout, lapack_int n, lapack_int ilo, lapack_int ihi,
                          double *a, lapack_int lda, double *tau) {
    int info = 0;
    int lwork = -1;
    double work_query = 0.0;
    dgehrd_(&n, &ilo, &ihi, a, &lda, tau, &work_query, &lwork, &info);
    lwork = static_cast<int>(work_query);
    array<double> work(std::max(1, lwork));
    dgehrd_(&n, &ilo, &ihi, a, &lda, tau, work.data(), &lwork, &info);
    return info;
}

inline int LAPACKE_dorghr(int matrix_layout, lapack_int n, lapack_int ilo, lapack_int ihi,
                          double *a, lapack_int lda, const double *tau) {
    int info = 0;
    int lwork = -1;
    double work_query = 0.0;
    dorghr_(&n, &ilo, &ihi, a, &lda, tau, &work_query, &lwork, &info);
    lwork = static_cast<int>(work_query);
    array<double> work(std::max(1, lwork));
    dorghr_(&n, &ilo, &ihi, a, &lda, tau, work.data(), &lwork, &info);
    return info;
}

inline int LAPACKE_dgesdd(int matrix_layout, char jobz, lapack_int m, lapack_int n, double *a,
                          lapack_int ldu, double *s, double *u, lapack_int ldu_unused, double *vt,
                          lapack_int ldvt) {
    int info = 0;
    int lwork = -1;
    double work_query = 0.0;
    array<int> iwork(8 * std::min(m, n));
    dgesdd_(&jobz, &n, &m, a, &ldu, s, vt, &ldvt, u, &ldu, &work_query, &lwork, iwork.data(),
            &info);
    lwork = static_cast<int>(work_query);
    array<double> work(std::max(1, lwork));
    dgesdd_(&jobz, &n, &m, a, &ldu, s, vt, &ldvt, u, &ldu, work.data(), &lwork, iwork.data(),
            &info);
    return info;
}

inline int LAPACKE_dsyevd(int matrix_layout, char jobz, char uplo, lapack_int n, double *a,
                          lapack_int lda, double *w) {
    int info = 0;
    int lwork = -1;
    int liwork = -1;
    double work_query = 0.0;
    int iwork_query = 0;
    dsyevd_(&jobz, &uplo, &n, a, &lda, w, &work_query, &lwork, &iwork_query, &liwork, &info);
    lwork = static_cast<int>(work_query);
    liwork = iwork_query;
    array<double> work(std::max(1, lwork));
    array<int> iwork(std::max(1, liwork));
    dsyevd_(&jobz, &uplo, &n, a, &lda, w, work.data(), &lwork, iwork.data(), &liwork, &info);
    return info;
}

inline int LAPACKE_dgtsv(int matrix_layout, lapack_int n, lapack_int nrhs, double *dl, double *d,
                         double *du, double *b, lapack_int ldb) {
    int info = 0;
    dgtsv_(&n, &nrhs, dl, d, du, b, &ldb, &info);
    return info;
}

inline int LAPACKE_dpotrf(int matrix_layout, char uplo, lapack_int n, double *a, lapack_int lda) {
    int info = 0;
    dpotrf_(&uplo, &n, a, &lda, &info);
    return info;
}

inline int LAPACKE_dpotrs(int matrix_layout, char uplo, lapack_int n, lapack_int nrhs,
                          const double *a, lapack_int lda, double *b, lapack_int ldb) {
    int info = 0;
    dpotrs_(&uplo, &n, &nrhs, a, &lda, b, &ldb, &info);
    return info;
}

inline int LAPACKE_dgetrs(int matrix_layout, char trans, lapack_int n, lapack_int nrhs,
                          const double *a, lapack_int lda, const lapack_int *ipiv, double *b,
                          lapack_int ldb) {
    int info = 0;
    dgetrs_(&trans, &n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
    return info;
}

inline int LAPACKE_dgetri(int matrix_layout, lapack_int n, double *a, lapack_int lda,
                          const lapack_int *ipiv) {
    int info = 0;
    int lwork = -1;
    double work_query = 0.0;
    dgetri_(&n, a, &lda, ipiv, &work_query, &lwork, &info);
    lwork = static_cast<int>(work_query);
    array<double> work(std::max(1, lwork));
    dgetri_(&n, a, &lda, ipiv, work.data(), &lwork, &info);
    return info;
}

inline int LAPACKE_dpotri(int matrix_layout, char uplo, lapack_int n, double *a, lapack_int lda) {
    int info = 0;
    dpotri_(&uplo, &n, a, &lda, &info);
    return info;
}

inline int LAPACKE_dgbtrf(int matrix_layout, lapack_int m, lapack_int n, lapack_int kl, lapack_int ku,
                          double *ab, lapack_int ldab, lapack_int *ipiv) {
    int info = 0;
    dgbtrf_(&m, &n, &kl, &ku, ab, &ldab, ipiv, &info);
    return info;
}

inline int LAPACKE_dgbtrs(int matrix_layout, char trans, lapack_int n, lapack_int kl, lapack_int ku,
                          lapack_int nrhs, const double *ab, lapack_int ldab, const lapack_int *ipiv,
                          double *b, lapack_int ldb) {
    int info = 0;
    dgbtrs_(&trans, &n, &kl, &ku, &nrhs, ab, &ldab, ipiv, b, &ldb, &info);
    return info;
}

inline int LAPACKE_dgeev(int matrix_layout, char jobvl, char jobvr, lapack_int n, double *a,
                         lapack_int lda, double *wr, double *wi, double *vl, lapack_int ldvl,
                         double *vr, lapack_int ldvr) {
    int info = 0;
    int lwork = -1;
    double work_query = 0.0;
    dgeev_(&jobvl, &jobvr, &n, a, &lda, wr, wi, vl, &ldvl, vr, &ldvr, &work_query, &lwork, &info);
    lwork = static_cast<int>(work_query);
    array<double> work(std::max(1, lwork));
    dgeev_(&jobvl, &jobvr, &n, a, &lda, wr, wi, vl, &ldvl, vr, &ldvr, work.data(), &lwork, &info);
    return info;
}

inline int LAPACKE_dsygvd(int matrix_layout, lapack_int itype, char jobz, char uplo, lapack_int n,
                          double *a, lapack_int lda, double *b, lapack_int ldb, double *w) {
    int info = 0;
    int lwork = -1;
    int liwork = -1;
    double work_query = 0.0;
    int iwork_query = 0;
    dsygvd_(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, &work_query, &lwork, &iwork_query, &liwork, &info);
    lwork = static_cast<int>(work_query);
    liwork = iwork_query;
    array<double> work(std::max(1, lwork));
    array<int> iwork(std::max(1, liwork));
    dsygvd_(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, work.data(), &lwork, iwork.data(), &liwork, &info);
    return info;
}

#else
// Linux / Native LAPACKE
#include <lapacke.h>
#endif

#endif
