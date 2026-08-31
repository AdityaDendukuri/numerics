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
#include "container/vector.hpp"
#include <cstdint>

namespace num {
namespace spectral {

/// Available implementations for Fourier transforms.
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

/// @brief Compute unnormalized forward 1D Fast Fourier Transform \f$X_k = \sum_{j=0}^{n-1} x_j e^{-2\pi i j k / n}\f$.
///
/// Dispatches to FFTW3 when linked or in-tree SIMD / sequential Cooley-Tukey Radix-2 / Bluestein kernel.
///
/// @param in Input complex vector of length \f$n\f$.
/// @param out Output complex spectrum vector (must be sized to \f$n\f$).
/// @param b FFT backend tag (`FFTBackend::fftw`, `FFTBackend::simd`, `FFTBackend::seq`).
/// @see ifft, rfft, FFTPlan
void fft(const CVector &in, CVector &out, FFTBackend b = default_fft_backend);

/// @brief Compute unnormalized inverse 1D Fast Fourier Transform \f$x_j = \sum_{k=0}^{n-1} X_k e^{+2\pi i j k / n}\f$.
///
/// Note that dividing by \f$n\f$ is required to recover the original amplitude.
///
/// @param in Input complex spectrum of length \f$n\f$.
/// @param out Output reconstructed complex vector (size \f$n\f$).
/// @param b FFT backend tag.
/// @see fft, irfft
void ifft(const CVector &in, CVector &out, FFTBackend b = default_fft_backend);

/// @brief Compute nonredundant half-spectrum of a real input signal: \f$n \to \lfloor n/2 \rfloor + 1\f$ complex coefficients.
///
/// Exploits Hermitian conjugate symmetry \f$X_{n-k} = X_k^*\f$ for real inputs.
///
/// @param in Real input vector of size \f$n\f$.
/// @param out Output complex half-spectrum vector of size \f$n/2 + 1\f$.
/// @param b FFT backend tag.
/// @see irfft, fft
void rfft(const Vector &in, CVector &out, FFTBackend b = default_fft_backend);

/// @brief Reconstruct an \f$n\f$-point real signal from its nonredundant half-spectrum.
///
/// @param in Complex half-spectrum of length \f$n/2 + 1\f$.
/// @param n Target length of the reconstructed real signal.
/// @param out Output real signal vector (size \f$n\f$).
/// @param b FFT backend tag.
/// @see rfft, ifft
void irfft(const CVector &in, int n, Vector &out, FFTBackend b = default_fft_backend);

/// Backend interface owned by FFTPlan.
struct FFTPlanImpl {
    virtual ~FFTPlanImpl() = default;
    virtual void execute(const CVector &in, CVector &out) const = 0;
};

/// @brief Precomputed 1D complex FFT execution plan for repeated transforms.
///
/// Pre-allocates twiddle factors, trigonometric lookup tables, or FFTW plans to amortize setup costs across multiple transforms of identical length.
class FFTPlan {
  public:
    /// @brief Precompute an \f$n\f$-point forward or inverse complex transform plan.
    /// @param n Transform length.
    /// @param forward `true` for forward FFT (\f$e^{-i\omega}\f$), `false` for inverse FFT (\f$e^{+i\omega}\f$).
    /// @param b FFT backend tag.
    explicit FFTPlan(int n, bool forward = true, FFTBackend b = default_fft_backend);
    ~FFTPlan();

    FFTPlan(const FFTPlan &) = delete;
    FFTPlan &operator=(const FFTPlan &) = delete;

    FFTPlan(FFTPlan &&) noexcept;
    FFTPlan &operator=(FFTPlan &&) noexcept;

    /// @brief Execute planned transform on input buffer.
    /// @param in Input complex vector (must have length \f$n\f$).
    /// @param out Output complex vector (must have length \f$n\f$).
    void execute(const CVector &in, CVector &out) const;

    [[nodiscard]] int size() const { return n_; }
    [[nodiscard]] FFTBackend backend() const { return backend_; }

  private:
    int n_;
    FFTBackend backend_;
    std::unique_ptr<FFTPlanImpl> impl_;
};

} // namespace spectral
} // namespace num
