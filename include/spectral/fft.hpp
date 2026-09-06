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
enum class fft_backend : std::uint8_t {
    seq,
    simd,
    stdsimd,
    fftw,
};

inline constexpr fft_backend seq = fft_backend::seq;
inline constexpr fft_backend fftw = fft_backend::fftw;
inline constexpr fft_backend fft_simd = fft_backend::simd;
inline constexpr fft_backend fft_stdsimd = fft_backend::stdsimd;

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

inline constexpr fft_backend default_fft_backend =
#ifdef NUMERICS_HAS_FFTW
    fft_backend::fftw;
#elif defined(NUMERICS_HAS_AVX2) || defined(NUMERICS_HAS_NEON)
    fft_backend::simd;
#else
                fft_backend::seq;
#endif

/// @brief Compute unnormalized forward 1D Fast Fourier Transform \f$X_k = \sum_{j=0}^{n-1} x_j e^{-2\pi i j k / n}\f$.
///
/// Dispatches to FFTW3 when linked or in-tree SIMD / sequential Cooley-Tukey Radix-2 / Bluestein kernel.
///
/// @param in Input complex vector of length \f$n\f$.
/// @param out Output complex spectrum vector (must be sized to \f$n\f$).
/// @param b FFT backend tag (`fft_backend::fftw`, `fft_backend::simd`, `fft_backend::seq`).
/// @see ifft, rfft, fft_plan
void fft(const cvec &in, cvec &out, fft_backend b = default_fft_backend);

/// @brief Compute unnormalized inverse 1D Fast Fourier Transform \f$x_j = \sum_{k=0}^{n-1} X_k e^{+2\pi i j k / n}\f$.
///
/// Note that dividing by \f$n\f$ is required to recover the original amplitude.
///
/// @param in Input complex spectrum of length \f$n\f$.
/// @param out Output reconstructed complex vector (size \f$n\f$).
/// @param b FFT backend tag.
/// @see fft, irfft
void ifft(const cvec &in, cvec &out, fft_backend b = default_fft_backend);

/// @brief Compute nonredundant half-spectrum of a real input signal: \f$n \to \lfloor n/2 \rfloor + 1\f$ complex coefficients.
///
/// Exploits Hermitian conjugate symmetry \f$X_{n-k} = X_k^*\f$ for real inputs.
///
/// @param in Real input vector of size \f$n\f$.
/// @param out Output complex half-spectrum vector of size \f$n/2 + 1\f$.
/// @param b FFT backend tag.
/// @see irfft, fft
void rfft(const vec &in, cvec &out, fft_backend b = default_fft_backend);

/// @brief Reconstruct an \f$n\f$-point real signal from its nonredundant half-spectrum.
///
/// @param in Complex half-spectrum of length \f$n/2 + 1\f$.
/// @param n Target length of the reconstructed real signal.
/// @param out Output real signal vector (size \f$n\f$).
/// @param b FFT backend tag.
/// @see rfft, ifft
void irfft(const cvec &in, int n, vec &out, fft_backend b = default_fft_backend);

/// Backend interface owned by fft_plan.
struct fft_plan_impl {
    virtual ~fft_plan_impl() = default;
    virtual void execute(const cvec &in, cvec &out) const = 0;
};

/// @brief Precomputed 1D complex FFT execution plan for repeated transforms.
///
/// Pre-allocates twiddle factors, trigonometric lookup tables, or FFTW plans to amortize setup costs across multiple transforms of identical length.
class fft_plan {
  public:
    /// @brief Precompute an \f$n\f$-point forward or inverse complex transform plan.
    /// @param n Transform length.
    /// @param forward `true` for forward FFT (\f$e^{-i\omega}\f$), `false` for inverse FFT (\f$e^{+i\omega}\f$).
    /// @param b FFT backend tag.
    explicit fft_plan(int n, bool forward = true, fft_backend b = default_fft_backend);
    ~fft_plan();

    fft_plan(const fft_plan &) = delete;
    fft_plan &operator=(const fft_plan &) = delete;

    fft_plan(fft_plan &&) noexcept;
    fft_plan &operator=(fft_plan &&) noexcept;

    /// @brief Execute planned transform on input buffer.
    /// @param in Input complex vector (must have length \f$n\f$).
    /// @param out Output complex vector (must have length \f$n\f$).
    void execute(const cvec &in, cvec &out) const;

    [[nodiscard]] int size() const { return n_; }
    [[nodiscard]] fft_backend backend() const { return backend_; }

  private:
    int n_;
    fft_backend backend_;
    std::unique_ptr<fft_plan_impl> impl_;
};

} // namespace spectral
} // namespace num
