/// @file spectral/backends/opt/impl.hpp
/// @brief Handwritten AVX2 / NEON butterfly for the FFT.
///
/// Processes 2 complex butterflies per SIMD iteration using 256-bit (AVX2) or
/// 128-bit (NEON) registers.  Falls back to the seq scalar butterfly when
/// neither ISA extension is available at compile time.
///
/// AVX2 complex butterfly (2 pairs per __m256d):
///   u  = [ur0, ui0, ur1, ui1]
///   v  = [vr0, vi0, vr1, vi1]
///   w  = [wr0, wi0, wr1, wi1]
///
///   w_re = unpacklo(w, w)     -> [wr0, wr0, wr1, wr1]
///   w_im = unpackhi(w, w)     -> [wi0, wi0, wi1, wi1]
///   v_sw = permute(v, 0b0101) -> [vi0, vr0, vi1, vr1]  (swap re/im)
///   t    = addsub(v*w_re, v_sw*w_im)
///        = [vr0*wr0 - vi0*wi0,  vr0*wi0 + vi0*wr0,
///           vr1*wr1 - vi1*wi1,  vr1*wi1 + vi1*wr1]
///
/// NEON deinterleaves with vld2q_f64 (SoA load), multiplies component-wise,
/// and re-interleaves with vst2q_f64.
#pragma once
#include "spectral/fft.hpp"
#include "../seq/impl.hpp"
#include <cmath>
#include <stdexcept>
#include <vector>

#ifdef NUMERICS_HAS_AVX2
#  include <immintrin.h>
#endif
#ifdef NUMERICS_HAS_NEON
#  include <arm_neon.h>
#endif

namespace backends {
namespace opt {

static constexpr double TWO_PI = 6.283185307179586476925286766559;

// Platform butterfly implementations

#ifdef NUMERICS_HAS_AVX2

/// Butterfly on hlen complex pairs using AVX2 (2 pairs per iteration).
static inline void butterfly(num::cplx* __restrict__ u,
                              num::cplx* __restrict__ v,
                              const num::cplx* __restrict__ tw,
                              int hlen)
{
    auto* ud = reinterpret_cast<double*>(u);
    auto* vd = reinterpret_cast<double*>(v);
    auto* wd = reinterpret_cast<const double*>(tw);

    int j = 0;
    for (; j + 1 < hlen; j += 2) {
        __m256d U   = _mm256_loadu_pd(ud + 2*j);
        __m256d V   = _mm256_loadu_pd(vd + 2*j);
        __m256d W   = _mm256_loadu_pd(wd + 2*j);
        // complex multiply V * W
        __m256d Wre = _mm256_unpacklo_pd(W, W);      // [wr0,wr0,wr1,wr1]
        __m256d Wim = _mm256_unpackhi_pd(W, W);      // [wi0,wi0,wi1,wi1]
        __m256d Vsw = _mm256_permute_pd(V, 0x5);     // [vi0,vr0,vi1,vr1]
        __m256d T   = _mm256_addsub_pd(
                          _mm256_mul_pd(V,   Wre),
                          _mm256_mul_pd(Vsw, Wim));
        _mm256_storeu_pd(ud + 2*j, _mm256_add_pd(U, T));
        _mm256_storeu_pd(vd + 2*j, _mm256_sub_pd(U, T));
    }
    // scalar tail for odd hlen
    for (; j < hlen; ++j) {
        num::cplx t = v[j] * tw[j];
        num::cplx uu = u[j];
        u[j] = uu + t;
        v[j] = uu - t;
    }
}

#elif defined(NUMERICS_HAS_NEON)

/// Butterfly on hlen complex pairs using NEON vld2/vst2 (SoA deinterleave).
static inline void butterfly(num::cplx* __restrict__ u,
                              num::cplx* __restrict__ v,
                              const num::cplx* __restrict__ tw,
                              int hlen)
{
    auto* ud = reinterpret_cast<double*>(u);
    auto* vd = reinterpret_cast<double*>(v);
    auto* wd = reinterpret_cast<const double*>(tw);

    int j = 0;
    for (; j + 1 < hlen; j += 2) {
        // Deinterleaved load: .val[0] = [re0,re1], .val[1] = [im0,im1]
        float64x2x2_t U = vld2q_f64(ud + 2*j);
        float64x2x2_t V = vld2q_f64(vd + 2*j);
        float64x2x2_t W = vld2q_f64(wd + 2*j);

        // T = V * W   (complex multiply, component-wise on SoA data)
        // Tr = Vr*Wr - Vi*Wi
        float64x2_t Tr = vfmsq_f64(vmulq_f64(V.val[0], W.val[0]),
                                     V.val[1], W.val[1]);
        // Ti = Vr*Wi + Vi*Wr
        float64x2_t Ti = vfmaq_f64(vmulq_f64(V.val[0], W.val[1]),
                                     V.val[1], W.val[0]);

        float64x2x2_t Ru, Rv;
        Ru.val[0] = vaddq_f64(U.val[0], Tr);
        Ru.val[1] = vaddq_f64(U.val[1], Ti);
        Rv.val[0] = vsubq_f64(U.val[0], Tr);
        Rv.val[1] = vsubq_f64(U.val[1], Ti);
        vst2q_f64(ud + 2*j, Ru);
        vst2q_f64(vd + 2*j, Rv);
    }
    // scalar tail
    for (; j < hlen; ++j) {
        num::cplx t = v[j] * tw[j];
        num::cplx uu = u[j];
        u[j] = uu + t;
        v[j] = uu - t;
    }
}

#else

/// Scalar fallback when no SIMD ISA is available.
static inline void butterfly(num::cplx* u, num::cplx* v,
                              const num::cplx* tw, int hlen)
{
    for (int j = 0; j < hlen; ++j) {
        num::cplx t = v[j] * tw[j];
        num::cplx uu = u[j];
        u[j] = uu + t;
        v[j] = uu - t;
    }
}

#endif // NUMERICS_HAS_AVX2 / NEON

// FFTPlanImpl

/// Precomputed twiddle factors + SIMD butterfly execution.
struct FFTPlanImpl {
    int  n;
    bool invert;
    std::vector<std::vector<num::cplx>> twiddles;

    FFTPlanImpl(int n_, bool inv) : n(n_), invert(inv) {
        if (n_ == 0 || (n_ & (n_ - 1)))
            throw std::invalid_argument("FFTPlan: length must be a power of two");
        for (int len = 2; len <= n_; len <<= 1) {
            double ang = TWO_PI / static_cast<double>(len) * (inv ? 1.0 : -1.0);
            num::cplx wlen{std::cos(ang), std::sin(ang)};
            std::vector<num::cplx> tw(len / 2);
            num::cplx w{1.0, 0.0};
            for (int j = 0; j < len / 2; ++j) { tw[j] = w; w *= wlen; }
            twiddles.push_back(std::move(tw));
        }
    }

    void execute(num::CVector& a) const {
        backends::seq::bit_reverse(a);
        num::cplx* data = a.data();
        int stage = 0;
        for (int len = 2; len <= n; len <<= 1, ++stage) {
            int hlen = len / 2;
            const num::cplx* tw = twiddles[stage].data();
            for (int i = 0; i < n; i += len)
                butterfly(data + i, data + i + hlen, tw, hlen);
        }
    }
};

// One-shot functions

inline void fft(const num::CVector& in, num::CVector& out) {
    int n = static_cast<int>(in.size());
    for (int i = 0; i < n; ++i) out[i] = in[i];
    FFTPlanImpl plan(n, false);
    plan.execute(out);
}

inline void ifft(const num::CVector& in, num::CVector& out) {
    int n = static_cast<int>(in.size());
    for (int i = 0; i < n; ++i) out[i] = in[i];
    FFTPlanImpl plan(n, true);
    plan.execute(out);
}

inline void rfft(const num::Vector& in, num::CVector& out) {
    int n = static_cast<int>(in.size());
    num::CVector tmp(static_cast<num::idx>(n), num::cplx{0, 0});
    for (int i = 0; i < n; ++i) tmp[i] = {in[i], 0.0};
    FFTPlanImpl plan(n, false);
    plan.execute(tmp);
    for (int k = 0; k < n / 2 + 1; ++k) out[k] = tmp[k];
}

inline void irfft(const num::CVector& in, int n, num::Vector& out) {
    num::CVector tmp(static_cast<num::idx>(n), num::cplx{0, 0});
    for (int k = 0; k < n / 2 + 1; ++k) tmp[k] = in[k];
    for (int k = 1; k < (n - 1) / 2 + 1; ++k)
        tmp[n - k] = std::conj(in[k]);
    FFTPlanImpl plan(n, true);
    plan.execute(tmp);
    for (int i = 0; i < n; ++i) out[i] = tmp[i].real();
}

} // namespace opt
} // namespace backends
