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
};

/// @brief Backend: in-tree FFT.
inline constexpr fft_backend seq = fft_backend::seq;
/// @brief Backend: in-tree FFT with SIMD intrinsics.
inline constexpr fft_backend fft_simd = fft_backend::simd;
/// @brief Backend: in-tree FFT with `std::simd`.
inline constexpr fft_backend fft_stdsimd = fft_backend::stdsimd;

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

/// @brief Default FFT backend: the SIMD FFT when available, else the sequential one.
inline constexpr fft_backend default_fft_backend =
#if defined(NUMERICS_HAS_AVX2) || defined(NUMERICS_HAS_NEON)
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

// Definitions.

#include "spectral/backends/opt/impl.hpp"
#include "spectral/backends/seq/impl.hpp"
#ifdef NUMERICS_HAS_STD_SIMD
#include "spectral/backends/stdsimd/impl.hpp"
#endif
#include <stdexcept>

namespace num {
namespace spectral {

// One-shot dispatch

inline void fft(const vec<cplx> &in, vec<cplx> &out, fft_backend b) {
    if (out.size() != in.size()) {
        throw std::invalid_argument("fft: in and out must have the same size");
    }
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

inline void ifft(const vec<cplx> &in, vec<cplx> &out, fft_backend b) {
    if (out.size() != in.size()) {
        throw std::invalid_argument("ifft: in and out must have the same size");
    }
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

inline void rfft(const vec<real> &in, vec<cplx> &out, fft_backend b) {
    if (static_cast<int>(out.size()) != (static_cast<int>(in.size()) / 2) + 1) {
        throw std::invalid_argument("rfft: out must have size n/2+1");
    }
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

inline void irfft(const vec<cplx> &in, int n, vec<real> &out, fft_backend b) {
    if (static_cast<int>(in.size()) != (n / 2) + 1) {
        throw std::invalid_argument("irfft: in must have size n/2+1");
    }
    if (static_cast<int>(out.size()) != n) {
        throw std::invalid_argument("irfft: out must have size n");
    }
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

// fft_plan

inline fft_plan::fft_plan(int n, bool forward, fft_backend b) : n_(n), backend_(b) {
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

inline fft_plan::~fft_plan() = default;
inline fft_plan::fft_plan(fft_plan &&) noexcept = default;
inline fft_plan &fft_plan::operator=(fft_plan &&) noexcept = default;

inline void fft_plan::execute(const vec<cplx> &in, vec<cplx> &out) const {
    if (static_cast<int>(in.size()) != n_ || static_cast<int>(out.size()) != n_) {
        throw std::invalid_argument("fft_plan::execute: size mismatch");
    }
    impl_->execute(in, out);
}

} // namespace spectral
} // namespace num
