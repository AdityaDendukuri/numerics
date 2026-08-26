/// @file container/matrix_ops.hpp
/// @brief Level-2 and level-3 operations on dense matrices, dispatched at compile time.
///
/// One overload per backend tag, so selection resolves during compilation and the
/// chosen body inlines into the caller. The per-backend implementations keep their
/// original structure under `num::backends`; only the dispatch changed, from a
/// runtime switch in a translation unit to an overload set.
#pragma once

#include "container/matrix.hpp"
#include "container/vector_ops.hpp"
#include "core/policy.hpp"
#include "kernel/raw.hpp"
#include <algorithm>
#include <cstdio>

#ifdef NUMERICS_HAS_BLAS
#include <cblas.h>
#endif
#ifdef NUMERICS_HAS_CUDA
#include "container/parallel/cuda_ops.hpp"
#endif
#ifdef NUMERICS_HAS_AVX2
#include <immintrin.h>
#endif
#if defined(NUMERICS_HAS_NEON) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace num {

namespace backends {

/// One-shot notice when a BLAS path is requested but BLAS was not configured.
inline void warn_blas_unavailable() {
#ifndef NUMERICS_HAS_BLAS
    static bool warned = false;
    if (!warned) {
        warned = true;
        std::fprintf(stderr,
                     "[numerics] WARNING: backend::blas requested but BLAS was not found "
                     "at configure time.\n           Falling back to backend::seq.\n");
    }
#endif
}


namespace seq {
inline void matmul(const Matrix &A, const Matrix &B, Matrix &C) {
    const idx M = A.rows(), K = A.cols(), N = B.cols();
    for (idx i = 0; i < M; ++i) {
        for (idx j = 0; j < N; ++j) {
            C(i, j) = 0;
            for (idx k = 0; k < K; ++k) {
                C(i, j) += A(i, k) * B(k, j);
            }
        }
    }
}
inline void matvec(const Matrix &A, const Vector &x, Vector &y) {
    kernel::raw::matvec(y.data(), A.data(), x.data(), A.rows(), A.cols());
}
inline void matadd(real alpha, const Matrix &A, real beta, const Matrix &B, Matrix &C) {
    kernel::raw::axpbyz(C.data(), A.data(), B.data(), alpha, beta, A.size());
}
inline void matmul_blocked(const Matrix &A, const Matrix &B, Matrix &C, idx block_size) {
    const idx M = A.rows(), K = A.cols(), N = B.cols();
    std::fill_n(C.data(), M * N, real(0));

    for (idx ii = 0; ii < M; ii += block_size) {
        const idx i_end = std::min(ii + block_size, M);
        for (idx jj = 0; jj < N; jj += block_size) {
            const idx j_end = std::min(jj + block_size, N);
            for (idx kk = 0; kk < K; kk += block_size) {
                const idx k_end = std::min(kk + block_size, K);
                for (idx i = ii; i < i_end; ++i) {
                    for (idx k = kk; k < k_end; ++k) {
                        const real a_ik = A(i, k);
                        for (idx j = jj; j < j_end; ++j) {
                            C(i, j) += a_ik * B(k, j);
                        }
                    }
                }
            }
        }
    }
}
inline void matmul_register_blocked(const Matrix &A, const Matrix &B, Matrix &C, idx block_size,
                             idx reg_size) {
    const idx M = A.rows(), K = A.cols(), N = B.cols();
    std::fill_n(C.data(), M * N, real(0));

    for (idx ii = 0; ii < M; ii += block_size) {
        const idx i_lim = std::min(ii + block_size, M);
        for (idx jj = 0; jj < N; jj += block_size) {
            const idx j_lim = std::min(jj + block_size, N);
            for (idx kk = 0; kk < K; kk += block_size) {
                const idx k_lim = std::min(kk + block_size, K);
                for (idx ir = ii; ir < i_lim; ir += reg_size) {
                    const idx ri = std::min(ir + reg_size, i_lim);
                    for (idx jr = jj; jr < j_lim; jr += reg_size) {
                        const idx rj = std::min(jr + reg_size, j_lim);
                        real c[4][4] = {};
                        for (idx i = ir; i < ri; ++i) {
                            for (idx j = jr; j < rj; ++j) {
                                c[i - ir][j - jr] = C(i, j);
                            }
                        }
                        for (idx k = kk; k < k_lim; ++k) {
                            for (idx i = ir; i < ri; ++i) {
                                const real a_ik = A(i, k);
                                for (idx j = jr; j < rj; ++j) {
                                    c[i - ir][j - jr] += a_ik * B(k, j);
                                }
                            }
                        }
                        for (idx i = ir; i < ri; ++i) {
                            for (idx j = jr; j < rj; ++j) {
                                C(i, j) = c[i - ir][j - jr];
                            }
                        }
                    }
                }
            }
        }
    }
}
} // namespace seq

namespace blas {
inline void matmul(const Matrix &A, const Matrix &B, Matrix &C) {
    warn_blas_unavailable();
#ifdef NUMERICS_HAS_BLAS
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<int>(A.rows()),
                static_cast<int>(B.cols()), static_cast<int>(A.cols()), 1.0, A.data(),
                static_cast<int>(A.cols()), B.data(), static_cast<int>(B.cols()), 0.0, C.data(),
                static_cast<int>(C.cols()));
#else
    seq::matmul_blocked(A, B, C, 64);
#endif
}
inline void matvec(const Matrix &A, const Vector &x, Vector &y) {
    warn_blas_unavailable();
#ifdef NUMERICS_HAS_BLAS
    cblas_dgemv(CblasRowMajor, CblasNoTrans, static_cast<int>(A.rows()), static_cast<int>(A.cols()),
                1.0, A.data(), static_cast<int>(A.cols()), x.data(), 1, 0.0, y.data(), 1);
#else
    seq::matvec(A, x, y);
#endif
}
inline void matadd(real alpha, const Matrix &A, real beta, const Matrix &B, Matrix &C) {
    warn_blas_unavailable();
#ifdef NUMERICS_HAS_BLAS
    cblas_dcopy(static_cast<int>(A.size()), A.data(), 1, C.data(), 1);
    cblas_dscal(static_cast<int>(C.size()), alpha, C.data(), 1);
    cblas_daxpy(static_cast<int>(B.size()), beta, B.data(), 1, C.data(), 1);
#else
    seq::matadd(alpha, A, beta, B, C);
#endif
}
} // namespace blas

namespace omp {
inline void matmul(const Matrix &A, const Matrix &B, Matrix &C) {
#ifdef NUMERICS_HAS_OMP
    constexpr idx BS = 64;
    const idx M = A.rows(), K = A.cols(), N = B.cols();
    std::fill_n(C.data(), M * N, real(0));
#pragma omp parallel for schedule(dynamic) collapse(2)
    for (idx ii = 0; ii < M; ii += BS) {
        for (idx jj = 0; jj < N; jj += BS) {
            const idx i_lim = std::min(ii + BS, M);
            const idx j_lim = std::min(jj + BS, N);
            for (idx kk = 0; kk < K; kk += BS) {
                const idx k_lim = std::min(kk + BS, K);
                for (idx i = ii; i < i_lim; ++i) {
                    for (idx k = kk; k < k_lim; ++k) {
                        const real a_ik = A(i, k);
                        for (idx j = jj; j < j_lim; ++j) {
                            C(i, j) += a_ik * B(k, j);
                        }
                    }
                }
            }
        }
    }
#else
    seq::matmul(A, B, C);
#endif
}
inline void matvec(const Matrix &A, const Vector &x, Vector &y) {
#ifdef NUMERICS_HAS_OMP
#pragma omp parallel for schedule(static)
    for (idx i = 0; i < A.rows(); ++i) {
        real sum = 0;
        for (idx j = 0; j < A.cols(); ++j) {
            sum += A(i, j) * x[j];
        }
        y[i] = sum;
    }
#else
    seq::matvec(A, x, y);
#endif
}
inline void matadd(real alpha, const Matrix &A, real beta, const Matrix &B, Matrix &C) {
#ifdef NUMERICS_HAS_OMP
    const idx n = A.size();
#pragma omp parallel for schedule(static)
    for (idx i = 0; i < n; ++i) {
        C.data()[i] = (alpha * A.data()[i]) + (beta * B.data()[i]);
    }
#else
    seq::matadd(alpha, A, beta, B, C);
#endif
}
} // namespace omp

namespace gpu {
inline void matmul(const Matrix &A, const Matrix &B, Matrix &C) {
#ifdef NUMERICS_HAS_CUDA
    cuda::matmul(A.gpu_data(), B.gpu_data(), C.gpu_data(), A.rows(), A.cols(), B.cols());
#else
    seq::matmul(A, B, C);
#endif
}
inline void matvec(const Matrix &A, const Vector &x, Vector &y) {
#ifdef NUMERICS_HAS_CUDA
    cuda::matvec(A.gpu_data(), x.gpu_data(), y.gpu_data(), A.rows(), A.cols());
#else
    seq::matvec(A, x, y);
#endif
}
} // namespace gpu

namespace simd {

#ifdef NUMERICS_HAS_AVX2
static inline void avx_tile_4x4(const Matrix &A, const Matrix &B, Matrix &C, idx ir, idx jr, idx kk,
                                idx k_lim) {
    const idx N = B.cols();
    real *Crow = C.data() + ir * N;
    __m256d c0 = _mm256_loadu_pd(Crow + 0 * N + jr);
    __m256d c1 = _mm256_loadu_pd(Crow + 1 * N + jr);
    __m256d c2 = _mm256_loadu_pd(Crow + 2 * N + jr);
    __m256d c3 = _mm256_loadu_pd(Crow + 3 * N + jr);

    for (idx k = kk; k < k_lim; ++k) {
        __m256d b = _mm256_loadu_pd(B.data() + k * N + jr);
        c0 = _mm256_fmadd_pd(_mm256_set1_pd(A(ir + 0, k)), b, c0);
        c1 = _mm256_fmadd_pd(_mm256_set1_pd(A(ir + 1, k)), b, c1);
        c2 = _mm256_fmadd_pd(_mm256_set1_pd(A(ir + 2, k)), b, c2);
        c3 = _mm256_fmadd_pd(_mm256_set1_pd(A(ir + 3, k)), b, c3);
    }

    _mm256_storeu_pd(Crow + 0 * N + jr, c0);
    _mm256_storeu_pd(Crow + 1 * N + jr, c1);
    _mm256_storeu_pd(Crow + 2 * N + jr, c2);
    _mm256_storeu_pd(Crow + 3 * N + jr, c3);
}

inline void matmul_avx(const Matrix &A, const Matrix &B, Matrix &C, idx block_size) {
    const idx M = A.rows(), K = A.cols(), N = B.cols();
    std::fill_n(C.data(), M * N, real(0));
    for (idx ii = 0; ii < M; ii += block_size) {
        const idx i_lim = std::min(ii + block_size, M);
        for (idx jj = 0; jj < N; jj += block_size) {
            const idx j_lim = std::min(jj + block_size, N);
            for (idx kk = 0; kk < K; kk += block_size) {
                const idx k_lim = std::min(kk + block_size, K);
                idx ir = ii;
                for (; ir + 4 <= i_lim; ir += 4) {
                    idx jr = jj;
                    for (; jr + 4 <= j_lim; jr += 4)
                        avx_tile_4x4(A, B, C, ir, jr, kk, k_lim);
                    for (; jr < j_lim; ++jr) {
                        real c0 = C(ir + 0, jr), c1 = C(ir + 1, jr);
                        real c2 = C(ir + 2, jr), c3 = C(ir + 3, jr);
                        for (idx k = kk; k < k_lim; ++k) {
                            real b = B(k, jr);
                            c0 += A(ir + 0, k) * b;
                            c1 += A(ir + 1, k) * b;
                            c2 += A(ir + 2, k) * b;
                            c3 += A(ir + 3, k) * b;
                        }
                        C(ir + 0, jr) = c0;
                        C(ir + 1, jr) = c1;
                        C(ir + 2, jr) = c2;
                        C(ir + 3, jr) = c3;
                    }
                }
                for (; ir < i_lim; ++ir) {
                    for (idx k = kk; k < k_lim; ++k) {
                        const real a_ik = A(ir, k);
                        for (idx j = jj; j < j_lim; ++j)
                            C(ir, j) += a_ik * B(k, j);
                    }
                }
            }
        }
    }
}

inline void matvec_avx(const Matrix &A, const Vector &x, Vector &y) {
    const idx M = A.rows(), N = A.cols();
    for (idx i = 0; i < M; ++i) {
        __m256d acc = _mm256_setzero_pd();
        idx j = 0;
        for (; j + 4 <= N; j += 4) {
            __m256d a = _mm256_loadu_pd(A.data() + i * N + j);
            __m256d xv = _mm256_loadu_pd(x.data() + j);
            acc = _mm256_fmadd_pd(a, xv, acc);
        }
        __m128d lo = _mm256_castpd256_pd128(acc);
        __m128d hi = _mm256_extractf128_pd(acc, 1);
        __m128d sum = _mm_add_pd(lo, hi);
        sum = _mm_hadd_pd(sum, sum);
        real result = _mm_cvtsd_f64(sum);
        for (; j < N; ++j)
            result += A(i, j) * x[j];
        y[i] = result;
    }
}
#endif

#ifdef NUMERICS_HAS_NEON
static inline void neon_tile_4x4(const Matrix &A, const Matrix &B, Matrix &C, idx ir, idx jr,
                                 idx kk, idx k_lim) {
    const idx N = B.cols();
    real *Crow = C.data() + (ir * N);
    float64x2_t c0lo = vld1q_f64(Crow + 0 * N + jr);
    float64x2_t c0hi = vld1q_f64(Crow + 0 * N + jr + 2);
    float64x2_t c1lo = vld1q_f64(Crow + 1 * N + jr);
    float64x2_t c1hi = vld1q_f64(Crow + 1 * N + jr + 2);
    float64x2_t c2lo = vld1q_f64(Crow + 2 * N + jr);
    float64x2_t c2hi = vld1q_f64(Crow + 2 * N + jr + 2);
    float64x2_t c3lo = vld1q_f64(Crow + 3 * N + jr);
    float64x2_t c3hi = vld1q_f64(Crow + 3 * N + jr + 2);

    for (idx k = kk; k < k_lim; ++k) {
        const real *Brow = B.data() + (k * N) + jr;
        float64x2_t blo = vld1q_f64(Brow), bhi = vld1q_f64(Brow + 2);
        float64x2_t a0 = vdupq_n_f64(A(ir + 0, k)), a1 = vdupq_n_f64(A(ir + 1, k));
        float64x2_t a2 = vdupq_n_f64(A(ir + 2, k)), a3 = vdupq_n_f64(A(ir + 3, k));
        c0lo = vfmaq_f64(c0lo, a0, blo);
        c0hi = vfmaq_f64(c0hi, a0, bhi);
        c1lo = vfmaq_f64(c1lo, a1, blo);
        c1hi = vfmaq_f64(c1hi, a1, bhi);
        c2lo = vfmaq_f64(c2lo, a2, blo);
        c2hi = vfmaq_f64(c2hi, a2, bhi);
        c3lo = vfmaq_f64(c3lo, a3, blo);
        c3hi = vfmaq_f64(c3hi, a3, bhi);
    }

    vst1q_f64(Crow + 0 * N + jr, c0lo);
    vst1q_f64(Crow + 0 * N + jr + 2, c0hi);
    vst1q_f64(Crow + 1 * N + jr, c1lo);
    vst1q_f64(Crow + 1 * N + jr + 2, c1hi);
    vst1q_f64(Crow + 2 * N + jr, c2lo);
    vst1q_f64(Crow + 2 * N + jr + 2, c2hi);
    vst1q_f64(Crow + 3 * N + jr, c3lo);
    vst1q_f64(Crow + 3 * N + jr + 2, c3hi);
}

inline void matmul_neon(const Matrix &A, const Matrix &B, Matrix &C, idx block_size) {
    const idx M = A.rows(), K = A.cols(), N = B.cols();
    std::fill_n(C.data(), M * N, real(0));
    for (idx ii = 0; ii < M; ii += block_size) {
        const idx i_lim = std::min(ii + block_size, M);
        for (idx jj = 0; jj < N; jj += block_size) {
            const idx j_lim = std::min(jj + block_size, N);
            for (idx kk = 0; kk < K; kk += block_size) {
                const idx k_lim = std::min(kk + block_size, K);
                idx ir = ii;
                for (; ir + 4 <= i_lim; ir += 4) {
                    idx jr = jj;
                    for (; jr + 4 <= j_lim; jr += 4) {
                        neon_tile_4x4(A, B, C, ir, jr, kk, k_lim);
                    }
                    for (; jr < j_lim; ++jr) {
                        real c0 = C(ir + 0, jr), c1 = C(ir + 1, jr);
                        real c2 = C(ir + 2, jr), c3 = C(ir + 3, jr);
                        for (idx k = kk; k < k_lim; ++k) {
                            real b = B(k, jr);
                            c0 += A(ir + 0, k) * b;
                            c1 += A(ir + 1, k) * b;
                            c2 += A(ir + 2, k) * b;
                            c3 += A(ir + 3, k) * b;
                        }
                        C(ir + 0, jr) = c0;
                        C(ir + 1, jr) = c1;
                        C(ir + 2, jr) = c2;
                        C(ir + 3, jr) = c3;
                    }
                }
                for (; ir < i_lim; ++ir) {
                    for (idx k = kk; k < k_lim; ++k) {
                        const real a_ik = A(ir, k);
                        for (idx j = jj; j < j_lim; ++j) {
                            C(ir, j) += a_ik * B(k, j);
                        }
                    }
                }
            }
        }
    }
}

inline void matvec_neon(const Matrix &A, const Vector &x, Vector &y) {
    const idx M = A.rows(), N = A.cols();
    for (idx i = 0; i < M; ++i) {
        float64x2_t acc = vdupq_n_f64(0.0);
        idx j = 0;
        for (; j + 2 <= N; j += 2) {
            float64x2_t a = vld1q_f64(A.data() + i * N + j);
            float64x2_t xv = vld1q_f64(x.data() + j);
            acc = vfmaq_f64(acc, a, xv);
        }
        real result = vgetq_lane_f64(acc, 0) + vgetq_lane_f64(acc, 1);
        for (; j < N; ++j) {
            result += A(i, j) * x[j];
        }
        y[i] = result;
    }
}
#endif

inline void matmul(const Matrix &A, const Matrix &B, Matrix &C, idx block_size) {
#if defined(NUMERICS_HAS_AVX2)
    matmul_avx(A, B, C, block_size);
#elif defined(NUMERICS_HAS_NEON)
    matmul_neon(A, B, C, block_size);
#else
    seq::matmul_blocked(A, B, C, block_size);
#endif
}

inline void matvec(const Matrix &A, const Vector &x, Vector &y) {
#if defined(NUMERICS_HAS_AVX2)
    matvec_avx(A, x, y);
#elif defined(NUMERICS_HAS_NEON)
    matvec_neon(A, x, y);
#else
    seq::matvec(A, x, y);
#endif
}

} // namespace simd

} // namespace backends


// -----------------------------------------------------------------------------
// Compile-time dispatch
// -----------------------------------------------------------------------------

inline void matvec(const Matrix &A, const Vector &x, Vector &y, backend::seq_t) {
    backends::seq::matvec(A, x, y);
}
inline void matvec(const Matrix &A, const Vector &x, Vector &y, backend::blocked_t) {
    backends::seq::matvec(A, x, y);
}
inline void matvec(const Matrix &A, const Vector &x, Vector &y, backend::simd_t) {
    backends::simd::matvec(A, x, y);
}
inline void matvec(const Matrix &A, const Vector &x, Vector &y, backend::blas_t) {
    backends::blas::matvec(A, x, y);
}
inline void matvec(const Matrix &A, const Vector &x, Vector &y, backend::lapack_t) {
    backends::blas::matvec(A, x, y);
}
inline void matvec(const Matrix &A, const Vector &x, Vector &y, backend::omp_t) {
    backends::omp::matvec(A, x, y);
}
inline void matvec(const Matrix &A, const Vector &x, Vector &y, backend::gpu_t) {
    backends::gpu::matvec(A, x, y);
}
inline void matvec(const Matrix &A, const Vector &x, Vector &y) {
    matvec(A, x, y, backend::dflt);
}

inline void matmul(const Matrix &A, const Matrix &B, Matrix &C, backend::seq_t) {
    backends::seq::matmul(A, B, C);
}
inline void matmul(const Matrix &A, const Matrix &B, Matrix &C, backend::blocked_t) {
    backends::seq::matmul_blocked(A, B, C, 64);
}
inline void matmul(const Matrix &A, const Matrix &B, Matrix &C, backend::simd_t) {
    backends::simd::matmul(A, B, C, 64);
}
inline void matmul(const Matrix &A, const Matrix &B, Matrix &C, backend::blas_t) {
    backends::blas::matmul(A, B, C);
}
inline void matmul(const Matrix &A, const Matrix &B, Matrix &C, backend::lapack_t) {
    backends::blas::matmul(A, B, C);
}
inline void matmul(const Matrix &A, const Matrix &B, Matrix &C, backend::omp_t) {
    backends::omp::matmul(A, B, C);
}
inline void matmul(const Matrix &A, const Matrix &B, Matrix &C, backend::gpu_t) {
    backends::gpu::matmul(A, B, C);
}
inline void matmul(const Matrix &A, const Matrix &B, Matrix &C) {
    matmul(A, B, C, backend::dflt);
}

inline void matadd(real alpha, const Matrix &A, real beta, const Matrix &B, Matrix &C,
                   backend::omp_t) {
    backends::omp::matadd(alpha, A, beta, B, C);
}
template <class Tag>
inline void matadd(real alpha, const Matrix &A, real beta, const Matrix &B, Matrix &C, Tag) {
    backends::seq::matadd(alpha, A, beta, B, C);
}
inline void matadd(real alpha, const Matrix &A, real beta, const Matrix &B, Matrix &C) {
    matadd(alpha, A, beta, B, C, backend::dflt);
}

/// @brief Cache-blocked product; tile edge chosen so the working set fits L2.
inline void matmul_blocked(const Matrix &A, const Matrix &B, Matrix &C, idx block_size = 64) {
    backends::seq::matmul_blocked(A, B, C, block_size);
}

/// @brief Cache blocking with an additional register tile inside each cache tile.
inline void matmul_register_blocked(const Matrix &A, const Matrix &B, Matrix &C,
                                    idx block_size = 64, idx reg_size = 4) {
    backends::seq::matmul_register_blocked(A, B, C, block_size, reg_size);
}

/// @brief SIMD product: AVX2+FMA on x86, NEON on AArch64, blocked otherwise.
inline void matmul_simd(const Matrix &A, const Matrix &B, Matrix &C, idx block_size = 64) {
    backends::simd::matmul(A, B, C, block_size);
}

/// @brief SIMD matrix-vector product.
inline void matvec_simd(const Matrix &A, const Vector &x, Vector &y) {
    backends::simd::matvec(A, x, y);
}

// -----------------------------------------------------------------------------
// Runtime bridge
// -----------------------------------------------------------------------------

inline void matvec(const Matrix &A, const Vector &x, Vector &y, Backend b) {
    with_backend(b, [&](auto tag) { matvec(A, x, y, tag); });
}

inline void matmul(const Matrix &A, const Matrix &B, Matrix &C, Backend b) {
    with_backend(b, [&](auto tag) { matmul(A, B, C, tag); });
}

inline void matadd(real alpha, const Matrix &A, real beta, const Matrix &B, Matrix &C, Backend b) {
    with_backend(b, [&](auto tag) { matadd(alpha, A, beta, B, C, tag); });
}

} // namespace num
