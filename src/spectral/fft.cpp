#include "spectral/fft.hpp"
#include "backends/fftw/impl.hpp"
#include "backends/opt/impl.hpp"
#include "backends/seq/impl.hpp"
#ifdef NUMERICS_HAS_STD_SIMD
#include "backends/stdsimd/impl.hpp"
#endif
#include <stdexcept>

namespace num {
namespace spectral {

// -- One-shot dispatch --------------------------------------------------------

void fft(const cvec &in, cvec &out, fft_backend b) {
    if (out.size() != in.size()) {
        throw std::invalid_argument("fft: in and out must have the same size");
    }
#ifdef NUMERICS_HAS_FFTW
    if (b == fft_backend::fftw) {
        num::backends::fftw::fft(in, out);
        return;
    }
#endif
#if defined(NUMERICS_HAS_AVX2) || defined(NUMERICS_HAS_NEON)
    if (b == fft_backend::simd) {
        num::backends::opt::fft(in, out);
        return;
    }
#endif
#ifdef NUMERICS_HAS_STD_SIMD
    if (b == fft_backend::stdsimd) {
        num::backends::stdsimd::fft(in, out);
        return;
    }
#endif
    // seq is the fallback for simd/stdsimd on unsupported platforms
    num::backends::seq::fft(in, out);
}

void ifft(const cvec &in, cvec &out, fft_backend b) {
    if (out.size() != in.size()) {
        throw std::invalid_argument("ifft: in and out must have the same size");
    }
#ifdef NUMERICS_HAS_FFTW
    if (b == fft_backend::fftw) {
        num::backends::fftw::ifft(in, out);
        return;
    }
#endif
#if defined(NUMERICS_HAS_AVX2) || defined(NUMERICS_HAS_NEON)
    if (b == fft_backend::simd) {
        num::backends::opt::ifft(in, out);
        return;
    }
#endif
#ifdef NUMERICS_HAS_STD_SIMD
    if (b == fft_backend::stdsimd) {
        num::backends::stdsimd::ifft(in, out);
        return;
    }
#endif
    num::backends::seq::ifft(in, out);
}

void rfft(const vec &in, cvec &out, fft_backend b) {
    if (static_cast<int>(out.size()) != (static_cast<int>(in.size()) / 2) + 1) {
        throw std::invalid_argument("rfft: out must have size n/2+1");
    }
#ifdef NUMERICS_HAS_FFTW
    if (b == fft_backend::fftw) {
        num::backends::fftw::rfft(in, out);
        return;
    }
#endif
#if defined(NUMERICS_HAS_AVX2) || defined(NUMERICS_HAS_NEON)
    if (b == fft_backend::simd) {
        num::backends::opt::rfft(in, out);
        return;
    }
#endif
#ifdef NUMERICS_HAS_STD_SIMD
    if (b == fft_backend::stdsimd) {
        num::backends::stdsimd::rfft(in, out);
        return;
    }
#endif
    num::backends::seq::rfft(in, out);
}

void irfft(const cvec &in, int n, vec &out, fft_backend b) {
    if (static_cast<int>(in.size()) != (n / 2) + 1) {
        throw std::invalid_argument("irfft: in must have size n/2+1");
    }
    if (static_cast<int>(out.size()) != n) {
        throw std::invalid_argument("irfft: out must have size n");
    }
#ifdef NUMERICS_HAS_FFTW
    if (b == fft_backend::fftw) {
        num::backends::fftw::irfft(in, n, out);
        return;
    }
#endif
#if defined(NUMERICS_HAS_AVX2) || defined(NUMERICS_HAS_NEON)
    if (b == fft_backend::simd) {
        num::backends::opt::irfft(in, n, out);
        return;
    }
#endif
#ifdef NUMERICS_HAS_STD_SIMD
    if (b == fft_backend::stdsimd) {
        num::backends::stdsimd::irfft(in, n, out);
        return;
    }
#endif
    num::backends::seq::irfft(in, n, out);
}

// -- fft_plan ------------------------------------------------------------------

fft_plan::fft_plan(int n, bool forward, fft_backend b) : n_(n), backend_(b) {
#ifdef NUMERICS_HAS_FFTW
    if (b == fft_backend::fftw) {
        impl_ = std::make_unique<num::backends::fftw::fft_plan_impl>(n, forward);
        return;
    }
#endif
#if defined(NUMERICS_HAS_AVX2) || defined(NUMERICS_HAS_NEON)
    if (b == fft_backend::simd) {
        impl_ = std::make_unique<num::backends::opt::fft_plan_impl>(n, !forward);
        return;
    }
#endif
#ifdef NUMERICS_HAS_STD_SIMD
    if (b == fft_backend::stdsimd) {
        impl_ = std::make_unique<num::backends::stdsimd::fft_plan_impl>(n, !forward);
        return;
    }
#endif
    impl_ = std::make_unique<num::backends::seq::fft_plan_impl>(n, !forward);
}

fft_plan::~fft_plan() = default;
fft_plan::fft_plan(fft_plan &&) noexcept = default;
fft_plan &fft_plan::operator=(fft_plan &&) noexcept = default;

void fft_plan::execute(const cvec &in, cvec &out) const {
    if (static_cast<int>(in.size()) != n_ || static_cast<int>(out.size()) != n_) {
        throw std::invalid_argument("fft_plan::execute: size mismatch");
    }
    impl_->execute(in, out);
}

} // namespace spectral
} // namespace num
