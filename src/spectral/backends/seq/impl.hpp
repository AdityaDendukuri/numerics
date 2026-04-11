/// @file spectral/backends/seq/impl.hpp
/// @brief Native Cooley-Tukey radix-2 DIT FFT (sequential, power-of-two sizes).
/// Only included by src/spectral/fft.cpp.
#pragma once
#include "spectral/fft.hpp"
#include <cmath>
#include <stdexcept>
#include <vector>

namespace backends {
namespace seq {

static constexpr double TWO_PI = 6.283185307179586476925286766559;

inline void bit_reverse(num::CVector& a) {
    num::idx n = a.size();
    for (num::idx i = 1, j = 0; i < n; ++i) {
        num::idx bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j)
            std::swap(a[i], a[j]);
    }
}

inline void cooley_tukey(num::CVector& a, bool invert) {
    num::idx n = a.size();
    if (n == 0 || (n & (n - 1)))
        throw std::invalid_argument("FFT: length must be a power of two");
    bit_reverse(a);
    for (num::idx len = 2; len <= n; len <<= 1) {
        double ang = TWO_PI / static_cast<double>(len) * (invert ? 1.0 : -1.0);
        num::cplx wlen{std::cos(ang), std::sin(ang)};
        for (num::idx i = 0; i < n; i += len) {
            num::cplx w{1.0, 0.0};
            for (num::idx j = 0; j < len / 2; ++j) {
                num::cplx u        = a[i + j];
                num::cplx v        = a[i + j + len / 2] * w;
                a[i + j]           = u + v;
                a[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
}

inline void fft(const num::CVector& in, num::CVector& out) {
    for (num::idx i = 0; i < in.size(); ++i)
        out[i] = in[i];
    cooley_tukey(out, false);
}

inline void ifft(const num::CVector& in, num::CVector& out) {
    for (num::idx i = 0; i < in.size(); ++i)
        out[i] = in[i];
    cooley_tukey(out, true);
}

inline void rfft(const num::Vector& in, num::CVector& out) {
    num::idx     n = in.size();
    num::CVector tmp(n, num::cplx{0, 0});
    for (num::idx i = 0; i < n; ++i)
        tmp[i] = {in[i], 0.0};
    cooley_tukey(tmp, false);
    for (num::idx k = 0; k < n / 2 + 1; ++k)
        out[k] = tmp[k];
}

inline void irfft(const num::CVector& in, int n, num::Vector& out) {
    num::CVector tmp(static_cast<num::idx>(n), num::cplx{0, 0});
    for (num::idx k = 0; k < static_cast<num::idx>(n / 2 + 1); ++k)
        tmp[k] = in[k];
    for (num::idx k = 1; k < static_cast<num::idx>((n - 1) / 2 + 1); ++k)
        tmp[static_cast<num::idx>(n) - k] = std::conj(in[k]);
    cooley_tukey(tmp, true);
    for (num::idx i = 0; i < static_cast<num::idx>(n); ++i)
        out[i] = tmp[i].real();
}

// Precomputed plan: twiddle factors per butterfly stage
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
        bit_reverse(a);
        int stage = 0;
        for (int len = 2; len <= n; len <<= 1, ++stage) {
            const auto& tw = twiddles[stage];
            for (int i = 0; i < n; i += len) {
                for (int j = 0; j < len / 2; ++j) {
                    num::cplx u        = a[i + j];
                    num::cplx v        = a[i + j + len / 2] * tw[j];
                    a[i + j]           = u + v;
                    a[i + j + len / 2] = u - v;
                }
            }
        }
    }
};

} // namespace seq
} // namespace backends
