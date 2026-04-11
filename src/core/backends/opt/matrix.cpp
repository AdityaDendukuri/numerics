/// @file core/backends/opt/matrix.cpp
/// @brief SIMD backend  -- hand-written AVX2/NEON matmul and matvec
///
/// Compile-time dispatch:
///   NUMERICS_HAS_AVX2  -> AVX-256 + FMA  (x86-64, 4 doubles/register)
///   NUMERICS_HAS_NEON  -> ARM NEON       (AArch64, 2 doubles/register)
///   neither            -> falls back to cache-blocked scalar
///
/// Both backends use the same register-tile structure:
///   Outer cache tile: ii -> jj -> kk    (B tile stays in L2)
///   Inner reg tile:   4 rows x 4 cols  (AVX: 4 YMM regs; NEON: 8 Q-regs)
///   Hot k loop:       one vector FMA per row, zero loop overhead for j

#include "core/matrix.hpp"
#include "../seq/impl.hpp"
#include <algorithm>
#include <cassert>

#ifdef NUMERICS_HAS_AVX2
    #include <immintrin.h>
#endif

#ifdef NUMERICS_HAS_NEON
    #include <arm_neon.h>
#endif

namespace num::backends::simd {

static_assert(sizeof(real) == 8, "SIMD kernels require real == double");

// AVX-256 backend
#ifdef NUMERICS_HAS_AVX2

static inline void avx_tile_4x4(const Matrix& A,
                                const Matrix& B,
                                Matrix&       C,
                                idx           ir,
                                idx           jr,
                                idx           kk,
                                idx           k_lim) {
    const idx N    = B.cols();
    real*     Crow = C.data() + ir * N;

    __m256d c0 = _mm256_loadu_pd(Crow + 0 * N + jr);
    __m256d c1 = _mm256_loadu_pd(Crow + 1 * N + jr);
    __m256d c2 = _mm256_loadu_pd(Crow + 2 * N + jr);
    __m256d c3 = _mm256_loadu_pd(Crow + 3 * N + jr);

    for (idx k = kk; k < k_lim; ++k) {
        __m256d b = _mm256_loadu_pd(B.data() + k * N + jr);
        c0        = _mm256_fmadd_pd(_mm256_set1_pd(A(ir + 0, k)), b, c0);
        c1        = _mm256_fmadd_pd(_mm256_set1_pd(A(ir + 1, k)), b, c1);
        c2        = _mm256_fmadd_pd(_mm256_set1_pd(A(ir + 2, k)), b, c2);
        c3        = _mm256_fmadd_pd(_mm256_set1_pd(A(ir + 3, k)), b, c3);
    }

    _mm256_storeu_pd(Crow + 0 * N + jr, c0);
    _mm256_storeu_pd(Crow + 1 * N + jr, c1);
    _mm256_storeu_pd(Crow + 2 * N + jr, c2);
    _mm256_storeu_pd(Crow + 3 * N + jr, c3);
}

static void matmul_avx(const Matrix& A,
                       const Matrix& B,
                       Matrix&       C,
                       idx           block_size) {
    const idx M = A.rows(), K = A.cols(), N = B.cols();
    std::fill_n(C.data(), M * N, real(0));

    for (idx ii = 0; ii < M; ii += block_size) {
        const idx i_lim = std::min(ii + block_size, M);
        for (idx jj = 0; jj < N; jj += block_size) {
            const idx j_lim = std::min(jj + block_size, N);
            for (idx kk = 0; kk < K; kk += block_size) {
                const idx k_lim = std::min(kk + block_size, K);
                idx       ir    = ii;
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

static void matvec_avx(const Matrix& A, const Vector& x, Vector& y) {
    const idx M = A.rows(), N = A.cols();
    for (idx i = 0; i < M; ++i) {
        __m256d acc = _mm256_setzero_pd();
        idx     j   = 0;
        for (; j + 4 <= N; j += 4) {
            __m256d a  = _mm256_loadu_pd(A.data() + i * N + j);
            __m256d xv = _mm256_loadu_pd(x.data() + j);
            acc        = _mm256_fmadd_pd(a, xv, acc);
        }
        __m128d lo  = _mm256_castpd256_pd128(acc);
        __m128d hi  = _mm256_extractf128_pd(acc, 1);
        __m128d sum = _mm_add_pd(lo, hi);
        sum         = _mm_hadd_pd(sum, sum);
        real result = _mm_cvtsd_f64(sum);
        for (; j < N; ++j)
            result += A(i, j) * x[j];
        y[i] = result;
    }
}

#endif // NUMERICS_HAS_AVX2

// ARM NEON backend
#ifdef NUMERICS_HAS_NEON

static inline void neon_tile_4x4(const Matrix& A,
                                 const Matrix& B,
                                 Matrix&       C,
                                 idx           ir,
                                 idx           jr,
                                 idx           kk,
                                 idx           k_lim) {
    const idx N    = B.cols();
    real*     Crow = C.data() + ir * N;

    float64x2_t c0lo = vld1q_f64(Crow + 0 * N + jr);
    float64x2_t c0hi = vld1q_f64(Crow + 0 * N + jr + 2);
    float64x2_t c1lo = vld1q_f64(Crow + 1 * N + jr);
    float64x2_t c1hi = vld1q_f64(Crow + 1 * N + jr + 2);
    float64x2_t c2lo = vld1q_f64(Crow + 2 * N + jr);
    float64x2_t c2hi = vld1q_f64(Crow + 2 * N + jr + 2);
    float64x2_t c3lo = vld1q_f64(Crow + 3 * N + jr);
    float64x2_t c3hi = vld1q_f64(Crow + 3 * N + jr + 2);

    for (idx k = kk; k < k_lim; ++k) {
        const real* Brow = B.data() + k * N + jr;
        float64x2_t blo = vld1q_f64(Brow), bhi = vld1q_f64(Brow + 2);
        float64x2_t a0 = vdupq_n_f64(A(ir + 0, k)),
                    a1 = vdupq_n_f64(A(ir + 1, k));
        float64x2_t a2 = vdupq_n_f64(A(ir + 2, k)),
                    a3 = vdupq_n_f64(A(ir + 3, k));
        c0lo           = vfmaq_f64(c0lo, a0, blo);
        c0hi           = vfmaq_f64(c0hi, a0, bhi);
        c1lo           = vfmaq_f64(c1lo, a1, blo);
        c1hi           = vfmaq_f64(c1hi, a1, bhi);
        c2lo           = vfmaq_f64(c2lo, a2, blo);
        c2hi           = vfmaq_f64(c2hi, a2, bhi);
        c3lo           = vfmaq_f64(c3lo, a3, blo);
        c3hi           = vfmaq_f64(c3hi, a3, bhi);
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

static void matmul_neon(const Matrix& A,
                        const Matrix& B,
                        Matrix&       C,
                        idx           block_size) {
    const idx M = A.rows(), K = A.cols(), N = B.cols();
    std::fill_n(C.data(), M * N, real(0));

    for (idx ii = 0; ii < M; ii += block_size) {
        const idx i_lim = std::min(ii + block_size, M);
        for (idx jj = 0; jj < N; jj += block_size) {
            const idx j_lim = std::min(jj + block_size, N);
            for (idx kk = 0; kk < K; kk += block_size) {
                const idx k_lim = std::min(kk + block_size, K);
                idx       ir    = ii;
                for (; ir + 4 <= i_lim; ir += 4) {
                    idx jr = jj;
                    for (; jr + 4 <= j_lim; jr += 4)
                        neon_tile_4x4(A, B, C, ir, jr, kk, k_lim);
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

static void matvec_neon(const Matrix& A, const Vector& x, Vector& y) {
    const idx M = A.rows(), N = A.cols();
    for (idx i = 0; i < M; ++i) {
        float64x2_t acc = vdupq_n_f64(0.0);
        idx         j   = 0;
        for (; j + 2 <= N; j += 2) {
            float64x2_t a  = vld1q_f64(A.data() + i * N + j);
            float64x2_t xv = vld1q_f64(x.data() + j);
            acc            = vfmaq_f64(acc, a, xv);
        }
        real result = vgetq_lane_f64(acc, 0) + vgetq_lane_f64(acc, 1);
        for (; j < N; ++j)
            result += A(i, j) * x[j];
        y[i] = result;
    }
}

#endif // NUMERICS_HAS_NEON

// Implementations  -- compile-time dispatch to best available SIMD backend

void matmul(const Matrix& A, const Matrix& B, Matrix& C, idx block_size) {
#if defined(NUMERICS_HAS_AVX2)
    matmul_avx(A, B, C, block_size);
#elif defined(NUMERICS_HAS_NEON)
    matmul_neon(A, B, C, block_size);
#else
    num::backends::seq::matmul_blocked(A, B, C, block_size);
#endif
}

void matvec(const Matrix& A, const Vector& x, Vector& y) {
#if defined(NUMERICS_HAS_AVX2)
    matvec_avx(A, x, y);
#elif defined(NUMERICS_HAS_NEON)
    matvec_neon(A, x, y);
#else
    num::backends::seq::matvec(A, x, y);
#endif
}

} // namespace num::backends::simd
