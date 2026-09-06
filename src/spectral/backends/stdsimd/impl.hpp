/// @file spectral/backends/stdsimd/impl.hpp
/// @brief Experimental C++20 <experimental/simd> SIMD FFT backend.
/// Only included by src/spectral/fft.cpp when NUMERICS_HAS_STD_SIMD is defined.
#pragma once
#include <experimental/simd>

namespace num {
namespace backends {
namespace stdsimd {

namespace stdx = std::experimental;

static constexpr double TWO_PI = 6.283185307179586476925286766559;

struct fft_plan_impl : public num::spectral::fft_plan_impl {
    int n;
    bool invert;
    std::vector<std::vector<num::cplx>> twiddles;

    fft_plan_impl(int n_, bool inv) : n(n_), invert(inv) {
        if (n_ == 0 || (n_ & (n_ - 1)))
            throw std::invalid_argument("fft_plan: length must be a power of two");
        for (int len = 2; len <= n_; len <<= 1) {
            double ang = TWO_PI / static_cast<double>(len) * (inv ? 1.0 : -1.0);
            num::cplx wlen{std::cos(ang), std::sin(ang)};
            std::vector<num::cplx> tw(len / 2);
            num::cplx w{1.0, 0.0};
            for (int j = 0; j < len / 2; ++j) {
                tw[j] = w;
                w *= wlen;
            }
            twiddles.push_back(std::move(tw));
        }
    }

    void execute(const num::cvec &in, num::cvec &out) const override {
        using vd = stdx::simd<double, stdx::simd_abi::native<double>>;
        constexpr int W = static_cast<int>(vd::size());

        for (num::idx i = 0; i < static_cast<num::idx>(n); ++i)
            out[i] = in[i];
        backends::seq::bit_reverse(out);
        num::cplx *data = out.data();

        int stage = 0;
        for (int len = 2; len <= n; len <<= 1, ++stage) {
            int hlen = len / 2;
            const num::cplx *tw = twiddles[stage].data();

            for (int i = 0; i < n; i += len) {
                num::cplx *up = data + i;
                num::cplx *vp = data + i + hlen;

                int j = 0;
                for (; j + W <= hlen; j += W) {
                    vd ur([&](int k) -> double { return up[j + k].real(); });
                    vd ui([&](int k) -> double { return up[j + k].imag(); });
                    vd vr([&](int k) -> double { return vp[j + k].real(); });
                    vd vi([&](int k) -> double { return vp[j + k].imag(); });
                    vd wr([&](int k) -> double { return tw[j + k].real(); });
                    vd wi([&](int k) -> double { return tw[j + k].imag(); });

                    vd tr = vr * wr - vi * wi;
                    vd ti = vr * wi + vi * wr;

                    for (int k = 0; k < W; ++k) {
                        up[j + k] = {ur[k] + tr[k], ui[k] + ti[k]};
                        vp[j + k] = {ur[k] - tr[k], ui[k] - ti[k]};
                    }
                }
                for (; j < hlen; ++j) {
                    num::cplx t = vp[j] * tw[j];
                    num::cplx uu = up[j];
                    up[j] = uu + t;
                    vp[j] = uu - t;
                }
            }
        }
    }
};

inline void fft(const num::cvec &in, num::cvec &out) {
    int n = static_cast<int>(in.size());
    fft_plan_impl plan(n, false);
    plan.execute(in, out);
}

inline void ifft(const num::cvec &in, num::cvec &out) {
    int n = static_cast<int>(in.size());
    fft_plan_impl plan(n, true);
    plan.execute(in, out);
}

inline void rfft(const num::vec &in, num::cvec &out) {
    int n = static_cast<int>(in.size());
    num::cvec tmp(static_cast<num::idx>(n), num::cplx{0, 0});
    for (int i = 0; i < n; ++i)
        tmp[i] = {in[i], 0.0};
    num::cvec tmp_out(static_cast<num::idx>(n), num::cplx{0, 0});
    fft_plan_impl plan(n, false);
    plan.execute(tmp, tmp_out);
    for (int k = 0; k < n / 2 + 1; ++k)
        out[k] = tmp_out[k];
}

inline void irfft(const num::cvec &in, int n, num::vec &out) {
    num::cvec tmp(static_cast<num::idx>(n), num::cplx{0, 0});
    for (int k = 0; k < n / 2 + 1; ++k)
        tmp[k] = in[k];
    for (int k = 1; k < (n - 1) / 2 + 1; ++k)
        tmp[n - k] = std::conj(in[k]);
    num::cvec tmp_out(static_cast<num::idx>(n), num::cplx{0, 0});
    fft_plan_impl plan(n, true);
    plan.execute(tmp, tmp_out);
    for (int i = 0; i < n; ++i)
        out[i] = tmp_out[i].real();
}

} // namespace stdsimd
} // namespace backends
} // namespace num
