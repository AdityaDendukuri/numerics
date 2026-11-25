/// @file spectral/fft.hpp
/// @brief Fast Fourier transforms: \f$X_k = \sum_j x_j e^{-2\pi i j k / n}\f$.
/// Inverses are unnormalized.
#pragma once

#include "core/types.hpp"
#include "container/vector.hpp"
#include <cstdint>

namespace num {
namespace spectral {

/// @brief FFT backend.
enum class fft_backend : std::uint8_t {
    seq,
    simd,
    stdsimd,
    fftw,
};

/// @brief Backend: in-tree FFT.
inline constexpr fft_backend seq = fft_backend::seq;
/// @brief Backend: FFTW.
inline constexpr fft_backend fftw = fft_backend::fftw;
/// @brief Backend: in-tree FFT with SIMD intrinsics.
inline constexpr fft_backend fft_simd = fft_backend::simd;
/// @brief Backend: in-tree FFT with `std::simd`.
inline constexpr fft_backend fft_stdsimd = fft_backend::stdsimd;

/// @brief Whether the build links FFTW.
inline constexpr bool has_fftw =
#ifdef NUMERICS_HAS_FFTW
    true;
#else
    false;
#endif

/// @brief Whether the SIMD FFT is available.
inline constexpr bool has_fft_simd =
#if defined(NUMERICS_HAS_AVX2) || defined(NUMERICS_HAS_NEON)
    true;
#else
    false;
#endif

/// @brief Whether the `std::simd` FFT is available.
inline constexpr bool has_fft_stdsimd =
#ifdef NUMERICS_HAS_STD_SIMD
    true;
#else
    false;
#endif

/// @brief Default FFT backend: FFTW if linked, else the fastest in-tree one.
inline constexpr fft_backend default_fft_backend =
#ifdef NUMERICS_HAS_FFTW
    fft_backend::fftw;
#elif defined(NUMERICS_HAS_AVX2) || defined(NUMERICS_HAS_NEON)
    fft_backend::simd;
#else
                fft_backend::seq;
#endif

/// @brief Fast Fourier transform (FFT): \f$X_k = \sum_{j=0}^{n-1} x_j e^{-2\pi i j k / n}\f$.
void fft(const vec<cplx> &in, vec<cplx> &out, fft_backend b = default_fft_backend);

/// @brief Inverse FFT, unnormalized: \f$x_j = \sum_{k=0}^{n-1} X_k e^{2\pi i j k / n}\f$.
/// Divide by \f$n\f$ to recover the input.
void ifft(const vec<cplx> &in, vec<cplx> &out, fft_backend b = default_fft_backend);

/// @brief Real FFT: the \f$\lfloor n/2 \rfloor + 1\f$ nonredundant coefficients of real input.
void rfft(const vec<real> &in, vec<cplx> &out, fft_backend b = default_fft_backend);

/// @brief Inverse real FFT: an \f$n\f$-point real signal from its half spectrum.
void irfft(const vec<cplx> &in, int n, vec<real> &out, fft_backend b = default_fft_backend);

/// @brief FFT plan backend.
struct fft_plan_impl {
    virtual ~fft_plan_impl() = default;
    virtual void execute(const vec<cplx> &in, vec<cplx> &out) const = 0;
};

/// @brief FFT plan: a reusable transform of fixed length.
class fft_plan {
  public:
    /// @brief Plan for an \f$n\f$-point forward or inverse transform.
    explicit fft_plan(int n, bool forward = true, fft_backend b = default_fft_backend);
    ~fft_plan();

    fft_plan(const fft_plan &) = delete;
    fft_plan &operator=(const fft_plan &) = delete;

    fft_plan(fft_plan &&) noexcept;
    fft_plan &operator=(fft_plan &&) noexcept;

    /// @brief Execute the plan: \f$X \leftarrow \mathcal{F} x\f$, both of length \f$n\f$.
    void execute(const vec<cplx> &in, vec<cplx> &out) const;

    [[nodiscard]] int size() const { return n_; }
    [[nodiscard]] fft_backend backend() const { return backend_; }

  private:
    int n_;
    fft_backend backend_;
    std::unique_ptr<fft_plan_impl> impl_;
};

} // namespace spectral
} // namespace num
