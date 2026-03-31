/// @file spectral/backends/stdsimd/impl.hpp
/// @brief std::experimental::simd butterfly for FFT.
///
/// Uses the portable C++ SIMD abstraction (<experimental/simd>, GCC 11+).
/// On AVX2 platforms vd::size() == 4 (4 doubles/register), on NEON == 2.
///
/// The gather uses the generator-lambda constructor:
///   simd<double, abi> ur([](int k){ return a[j+k].real(); });
/// which the compiler maps to SIMD gather instructions when possible.
///
/// The scatter-store back to AoS is the main cost difference vs. the
/// handwritten backend -- the element-wise write loop is the comparison point.
///
/// Only compiled when NUMERICS_HAS_STD_SIMD is defined.
#pragma once
#ifdef NUMERICS_HAS_STD_SIMD
    #include "spectral/fft.hpp"
    #include "../seq/impl.hpp"
    #include <experimental/simd>
    #include <cmath>
    #include <stdexcept>
    #include <vector>

namespace stdx = std::experimental;

namespace backends {
namespace stdsimd {

static constexpr double TWO_PI = 6.283185307179586476925286766559;

// FFTPlanImpl

struct FFTPlanImpl {
    int                                 n;
    bool                                invert;
    std::vector<std::vector<num::cplx>> twiddles;

    FFTPlanImpl(int n_, bool inv)
        : n(n_)
        , invert(inv) {
        if (n_ == 0 || (n_ & (n_ - 1)))
            throw std::invalid_argument(
                "FFTPlan: length must be a power of two");
        for (int len = 2; len <= n_; len <<= 1) {
            double ang = TWO_PI / static_cast<double>(len) * (inv ? 1.0 : -1.0);
            num::cplx              wlen{std::cos(ang), std::sin(ang)};
            std::vector<num::cplx> tw(len / 2);
            num::cplx              w{1.0, 0.0};
            for (int j = 0; j < len / 2; ++j) {
                tw[j] = w;
                w *= wlen;
            }
            twiddles.push_back(std::move(tw));
        }
    }

    void execute(num::CVector& a) const {
        using vd        = stdx::simd<double, stdx::simd_abi::native<double>>;
        constexpr int W = static_cast<int>(vd::size());

        backends::seq::bit_reverse(a);
        num::cplx* data = a.data();

        int stage = 0;
        for (int len = 2; len <= n; len <<= 1, ++stage) {
            int              hlen = len / 2;
            const num::cplx* tw   = twiddles[stage].data();

            for (int i = 0; i < n; i += len) {
                num::cplx* up = data + i;
                num::cplx* vp = data + i + hlen;

                int j = 0;
                for (; j + W <= hlen; j += W) {
                    // Gather: split AoS complex into separate real/imag
                    // vectors.
                    vd ur([&](int k) -> double { return up[j + k].real(); });
                    vd ui([&](int k) -> double { return up[j + k].imag(); });
                    vd vr([&](int k) -> double { return vp[j + k].real(); });
                    vd vi([&](int k) -> double { return vp[j + k].imag(); });
                    vd wr([&](int k) -> double { return tw[j + k].real(); });
                    vd wi([&](int k) -> double { return tw[j + k].imag(); });

                    // Complex multiply: t = v * w
                    vd tr = vr * wr - vi * wi;
                    vd ti = vr * wi + vi * wr;

                    // Butterfly + scatter store
                    for (int k = 0; k < W; ++k) {
                        up[j + k] = {ur[k] + tr[k], ui[k] + ti[k]};
                        vp[j + k] = {ur[k] - tr[k], ui[k] - ti[k]};
                    }
                }
                // scalar tail
                for (; j < hlen; ++j) {
                    num::cplx t  = vp[j] * tw[j];
                    num::cplx uu = up[j];
                    up[j]        = uu + t;
                    vp[j]        = uu - t;
                }
            }
        }
    }
};

// One-shot functions

inline void fft(const num::CVector& in, num::CVector& out) {
    int n = static_cast<int>(in.size());
    for (int i = 0; i < n; ++i)
        out[i] = in[i];
    FFTPlanImpl plan(n, false);
    plan.execute(out);
}

inline void ifft(const num::CVector& in, num::CVector& out) {
    int n = static_cast<int>(in.size());
    for (int i = 0; i < n; ++i)
        out[i] = in[i];
    FFTPlanImpl plan(n, true);
    plan.execute(out);
}

inline void rfft(const num::Vector& in, num::CVector& out) {
    int          n = static_cast<int>(in.size());
    num::CVector tmp(static_cast<num::idx>(n), num::cplx{0, 0});
    for (int i = 0; i < n; ++i)
        tmp[i] = {in[i], 0.0};
    FFTPlanImpl plan(n, false);
    plan.execute(tmp);
    for (int k = 0; k < n / 2 + 1; ++k)
        out[k] = tmp[k];
}

inline void irfft(const num::CVector& in, int n, num::Vector& out) {
    num::CVector tmp(static_cast<num::idx>(n), num::cplx{0, 0});
    for (int k = 0; k < n / 2 + 1; ++k)
        tmp[k] = in[k];
    for (int k = 1; k < (n - 1) / 2 + 1; ++k)
        tmp[n - k] = std::conj(in[k]);
    FFTPlanImpl plan(n, true);
    plan.execute(tmp);
    for (int i = 0; i < n; ++i)
        out[i] = tmp[i].real();
}

} // namespace stdsimd
} // namespace backends

#endif // NUMERICS_HAS_STD_SIMD
