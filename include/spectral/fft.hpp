/// @file spectral/fft.hpp
/// @brief FFT interface with backend dispatch.
///
/// Each module defines its own backend enum for the choices relevant to it.
/// This enum covers spectral transforms; linalg uses num::Backend in core/policy.hpp.
///
/// Backends:
///   FFTBackend::seq   -- native Cooley-Tukey radix-2 DIT; always available.
///                        Input length must be a power of two.
///   FFTBackend::fftw  -- FFTW3 (AVX / NEON optimised); requires NUMERICS_HAS_FFTW.
///                        Falls back to FFTBackend::seq automatically if FFTW3 is absent.
///
/// Conventions (both backends):
///   Forward DFT:  X[k] = sum_{j=0}^{n-1} x[j] * exp(-2*pi*i*j*k/n)
///   Inverse DFT:  unnormalised (FFTW convention).  Divide by n to recover x.
///   rfft output:  n/2+1 complex bins  (Hermitian symmetry of real input).
#pragma once

#include "core/types.hpp"
#include "core/vector.hpp"

namespace num {
namespace spectral {

/// @brief Selects which backend handles an FFT operation.
enum class FFTBackend {
    seq,     ///< Native Cooley-Tukey radix-2 DIT -- always available (power-of-two only)
    simd,    ///< Handwritten AVX2 / NEON butterfly -- falls back to seq if unavailable
    stdsimd, ///< std::experimental::simd butterfly -- requires NUMERICS_HAS_STD_SIMD
    fftw,    ///< FFTW3 (mixed-radix, AVX / NEON optimised) -- requires NUMERICS_HAS_FFTW
};

// Convenience constants -- use these at call sites:
//   fft(in, out, num::spectral::fftw);
inline constexpr FFTBackend seq     = FFTBackend::seq;
inline constexpr FFTBackend fftw    = FFTBackend::fftw;
inline constexpr FFTBackend fft_simd    = FFTBackend::simd;
inline constexpr FFTBackend fft_stdsimd = FFTBackend::stdsimd;

/// True when FFTW3 was found at configure time.
inline constexpr bool has_fftw =
#ifdef NUMERICS_HAS_FFTW
    true;
#else
    false;
#endif

/// True when handwritten AVX2 or NEON backend is available.
inline constexpr bool has_fft_simd =
#if defined(NUMERICS_HAS_AVX2) || defined(NUMERICS_HAS_NEON)
    true;
#else
    false;
#endif

/// True when std::experimental::simd is available.
inline constexpr bool has_fft_stdsimd =
#ifdef NUMERICS_HAS_STD_SIMD
    true;
#else
    false;
#endif

/// Automatically selected at configure time: fftw > simd > seq.
inline constexpr FFTBackend default_fft_backend =
#ifdef NUMERICS_HAS_FFTW
    FFTBackend::fftw;
#elif defined(NUMERICS_HAS_AVX2) || defined(NUMERICS_HAS_NEON)
    FFTBackend::simd;
#else
    FFTBackend::seq;
#endif

// One-shot transforms

/// @brief Forward complex DFT.  out must be pre-allocated to in.size().
void fft(const CVector& in, CVector& out,
         FFTBackend b = default_fft_backend);

/// @brief Inverse complex DFT (unnormalised: result = n * true_inverse).
void ifft(const CVector& in, CVector& out,
          FFTBackend b = default_fft_backend);

/// @brief Real-to-complex forward DFT.  out must be pre-allocated to n/2+1.
void rfft(const Vector& in, CVector& out,
          FFTBackend b = default_fft_backend);

/// @brief Complex-to-real inverse DFT (unnormalised).
/// @param n  Length of the real output (must satisfy in.size() == n/2+1).
void irfft(const CVector& in, int n, Vector& out,
           FFTBackend b = default_fft_backend);

// Reusable plan

/// @brief Precomputed plan for repeated complex transforms of a fixed size.
///
/// @code
/// num::spectral::FFTPlan plan(1024);          // forward, default backend
/// for (auto& frame : frames)
///     plan.execute(frame, spectrum);           // O(n log n), no allocation
/// @endcode
class FFTPlan {
public:
    /// @param n        Transform size (must be power of two for FFTBackend::seq)
    /// @param forward  true = forward DFT, false = inverse DFT
    /// @param b        Backend to use (default: default_fft_backend)
    explicit FFTPlan(int n, bool forward = true,
                     FFTBackend b = default_fft_backend);
    ~FFTPlan();

    FFTPlan(const FFTPlan&)            = delete;
    FFTPlan& operator=(const FFTPlan&) = delete;

    void execute(const CVector& in, CVector& out) const;

    int        size()    const { return n_; }
    FFTBackend backend() const { return backend_; }

private:
    int        n_;
    FFTBackend backend_;
    void*      impl_;   // backends::seq::FFTPlanImpl or backends::fftw::FFTPlanImpl
};

} // namespace spectral
} // namespace num
