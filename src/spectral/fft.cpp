#include "spectral/fft.hpp"
#include "backends/seq/impl.hpp"
#include "backends/opt/impl.hpp"
#include "backends/fftw/impl.hpp"
#ifdef NUMERICS_HAS_STD_SIMD
#  include "backends/stdsimd/impl.hpp"
#endif
#include <stdexcept>

namespace num {
namespace spectral {

// -- One-shot dispatch --------------------------------------------------------

void fft(const CVector& in, CVector& out, FFTBackend b) {
    if (out.size() != in.size())
        throw std::invalid_argument("fft: in and out must have the same size");
#ifdef NUMERICS_HAS_FFTW
    if (b == FFTBackend::fftw)    { backends::fftw::fft(in, out);    return; }
#endif
#if defined(NUMERICS_HAS_AVX2) || defined(NUMERICS_HAS_NEON)
    if (b == FFTBackend::simd)    { backends::opt::fft(in, out);     return; }
#endif
#ifdef NUMERICS_HAS_STD_SIMD
    if (b == FFTBackend::stdsimd) { backends::stdsimd::fft(in, out); return; }
#endif
    // seq is the fallback for simd/stdsimd on unsupported platforms
    backends::seq::fft(in, out);
}

void ifft(const CVector& in, CVector& out, FFTBackend b) {
    if (out.size() != in.size())
        throw std::invalid_argument("ifft: in and out must have the same size");
#ifdef NUMERICS_HAS_FFTW
    if (b == FFTBackend::fftw)    { backends::fftw::ifft(in, out);    return; }
#endif
#if defined(NUMERICS_HAS_AVX2) || defined(NUMERICS_HAS_NEON)
    if (b == FFTBackend::simd)    { backends::opt::ifft(in, out);     return; }
#endif
#ifdef NUMERICS_HAS_STD_SIMD
    if (b == FFTBackend::stdsimd) { backends::stdsimd::ifft(in, out); return; }
#endif
    backends::seq::ifft(in, out);
}

void rfft(const Vector& in, CVector& out, FFTBackend b) {
    if (static_cast<int>(out.size()) != static_cast<int>(in.size()) / 2 + 1)
        throw std::invalid_argument("rfft: out must have size n/2+1");
#ifdef NUMERICS_HAS_FFTW
    if (b == FFTBackend::fftw)    { backends::fftw::rfft(in, out);    return; }
#endif
#if defined(NUMERICS_HAS_AVX2) || defined(NUMERICS_HAS_NEON)
    if (b == FFTBackend::simd)    { backends::opt::rfft(in, out);     return; }
#endif
#ifdef NUMERICS_HAS_STD_SIMD
    if (b == FFTBackend::stdsimd) { backends::stdsimd::rfft(in, out); return; }
#endif
    backends::seq::rfft(in, out);
}

void irfft(const CVector& in, int n, Vector& out, FFTBackend b) {
    if (static_cast<int>(in.size()) != n / 2 + 1)
        throw std::invalid_argument("irfft: in must have size n/2+1");
    if (static_cast<int>(out.size()) != n)
        throw std::invalid_argument("irfft: out must have size n");
#ifdef NUMERICS_HAS_FFTW
    if (b == FFTBackend::fftw)    { backends::fftw::irfft(in, n, out);    return; }
#endif
#if defined(NUMERICS_HAS_AVX2) || defined(NUMERICS_HAS_NEON)
    if (b == FFTBackend::simd)    { backends::opt::irfft(in, n, out);     return; }
#endif
#ifdef NUMERICS_HAS_STD_SIMD
    if (b == FFTBackend::stdsimd) { backends::stdsimd::irfft(in, n, out); return; }
#endif
    backends::seq::irfft(in, n, out);
}

// -- FFTPlan ------------------------------------------------------------------

FFTPlan::FFTPlan(int n, bool forward, FFTBackend b) : n_(n), backend_(b) {
#ifdef NUMERICS_HAS_FFTW
    if (b == FFTBackend::fftw) {
        impl_ = new backends::fftw::FFTPlanImpl(n, forward);
        return;
    }
#endif
#if defined(NUMERICS_HAS_AVX2) || defined(NUMERICS_HAS_NEON)
    if (b == FFTBackend::simd) {
        impl_ = new backends::opt::FFTPlanImpl(n, !forward);
        return;
    }
#endif
#ifdef NUMERICS_HAS_STD_SIMD
    if (b == FFTBackend::stdsimd) {
        impl_ = new backends::stdsimd::FFTPlanImpl(n, !forward);
        return;
    }
#endif
    impl_ = new backends::seq::FFTPlanImpl(n, !forward);
}

FFTPlan::~FFTPlan() {
#ifdef NUMERICS_HAS_FFTW
    if (backend_ == FFTBackend::fftw) {
        delete static_cast<backends::fftw::FFTPlanImpl*>(impl_);
        return;
    }
#endif
#if defined(NUMERICS_HAS_AVX2) || defined(NUMERICS_HAS_NEON)
    if (backend_ == FFTBackend::simd) {
        delete static_cast<backends::opt::FFTPlanImpl*>(impl_);
        return;
    }
#endif
#ifdef NUMERICS_HAS_STD_SIMD
    if (backend_ == FFTBackend::stdsimd) {
        delete static_cast<backends::stdsimd::FFTPlanImpl*>(impl_);
        return;
    }
#endif
    delete static_cast<backends::seq::FFTPlanImpl*>(impl_);
}

void FFTPlan::execute(const CVector& in, CVector& out) const {
    if (static_cast<int>(in.size()) != n_ || static_cast<int>(out.size()) != n_)
        throw std::invalid_argument("FFTPlan::execute: size mismatch");
#ifdef NUMERICS_HAS_FFTW
    if (backend_ == FFTBackend::fftw) {
        static_cast<backends::fftw::FFTPlanImpl*>(impl_)->execute(in, out);
        return;
    }
#endif
#if defined(NUMERICS_HAS_AVX2) || defined(NUMERICS_HAS_NEON)
    if (backend_ == FFTBackend::simd) {
        for (idx i = 0; i < static_cast<idx>(n_); ++i) out[i] = in[i];
        static_cast<backends::opt::FFTPlanImpl*>(impl_)->execute(out);
        return;
    }
#endif
#ifdef NUMERICS_HAS_STD_SIMD
    if (backend_ == FFTBackend::stdsimd) {
        for (idx i = 0; i < static_cast<idx>(n_); ++i) out[i] = in[i];
        static_cast<backends::stdsimd::FFTPlanImpl*>(impl_)->execute(out);
        return;
    }
#endif
    for (idx i = 0; i < static_cast<idx>(n_); ++i) out[i] = in[i];
    static_cast<backends::seq::FFTPlanImpl*>(impl_)->execute(out);
}

} // namespace spectral
} // namespace num
