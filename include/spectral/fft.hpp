/// @file spectral/fft.hpp
/// @brief FFT interface with backend dispatch.
///
/// Forward transform convention:
/// \f[
///   X_k=\sum_{j=0}^{n-1} x_j e^{-2\pi i jk/n}.
/// \f]
/// The inverse transform is unnormalized.
#pragma once

#include "core/types.hpp"
#include "core/vector.hpp"
#include <cstdint>

namespace num {
namespace spectral {

enum class FFTBackend : std::uint8_t {
  seq,
  simd,
  stdsimd,
  fftw,
};

inline constexpr FFTBackend seq = FFTBackend::seq;
inline constexpr FFTBackend fftw = FFTBackend::fftw;
inline constexpr FFTBackend fft_simd = FFTBackend::simd;
inline constexpr FFTBackend fft_stdsimd = FFTBackend::stdsimd;

inline constexpr bool has_fftw =
#ifdef NUMERICS_HAS_FFTW
  true;
#else
  false;
#endif

inline constexpr bool has_fft_simd =
#if defined(NUMERICS_HAS_AVX2) || defined(NUMERICS_HAS_NEON)
  true;
#else
  false;
#endif

inline constexpr bool has_fft_stdsimd =
#ifdef NUMERICS_HAS_STD_SIMD
  true;
#else
  false;
#endif

inline constexpr FFTBackend default_fft_backend =
#ifdef NUMERICS_HAS_FFTW
  FFTBackend::fftw;
#elif defined(NUMERICS_HAS_AVX2) || defined(NUMERICS_HAS_NEON)
  FFTBackend::simd;
#else
        FFTBackend::seq;
#endif

void fft(const CVector& in, CVector& out, FFTBackend b = default_fft_backend);

void ifft(const CVector& in, CVector& out, FFTBackend b = default_fft_backend);

void rfft(const Vector& in, CVector& out, FFTBackend b = default_fft_backend);

void irfft(const CVector& in, int n, Vector& out, FFTBackend b = default_fft_backend);

struct FFTPlanImpl {
  virtual ~FFTPlanImpl() = default;
  virtual void execute(const CVector& in, CVector& out) const = 0;
};

/// @brief Precomputed complex transform plan.

class FFTPlan {
public:
  explicit FFTPlan(int n, bool forward = true, FFTBackend b = default_fft_backend);
  ~FFTPlan();

  FFTPlan(const FFTPlan&) = delete;
  FFTPlan& operator=(const FFTPlan&) = delete;

  FFTPlan(FFTPlan&&) noexcept;
  FFTPlan& operator=(FFTPlan&&) noexcept;

  void execute(const CVector& in, CVector& out) const;

  [[nodiscard]] int size() const { return n_; }
  [[nodiscard]] FFTBackend backend() const { return backend_; }

private:
  int n_;
  FFTBackend backend_;
  std::unique_ptr<FFTPlanImpl> impl_;
};


} // namespace spectral
} // namespace num
