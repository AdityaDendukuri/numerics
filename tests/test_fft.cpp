/// @file test_fft.cpp
/// @brief Unit tests for num::spectral FFT module.
///
/// Tests cover:
///   - DC / single-frequency spikes (both backends)
///   - Round-trip: ifft(fft(x)) / n == x
///   - Round-trip: irfft(rfft(x)) / n == x
///   - Parseval identity: sum|X|^2 == n * sum|x|^2
///   - Linearity: fft(a*x + b*y) == a*fft(x) + b*fft(y)
///   - FFTPlan: matches one-shot fft, inverse plan round-trip
///   - Error handling: mismatched sizes throw invalid_argument
///   - FFTW and seq backends produce identical results (when FFTW available)

#include <gtest/gtest.h>
#include "spectral/fft.hpp"
#include "core/types.hpp"
#include <cmath>
#include <complex>
#include <functional>
#include <stdexcept>

using namespace num;
using namespace num::spectral;
using cplx = std::complex<double>;

static constexpr double TWO_PI = 6.283185307179586476925286766559;

// ---- helpers ----------------------------------------------------------------

static CVector make_cvec(int n, cplx val = cplx{}) {
    CVector v(static_cast<idx>(n));
    for (idx i = 0; i < static_cast<idx>(n); ++i) v[i] = val;
    return v;
}

static Vector make_vec(int n, real val = 0.0) {
    Vector v(static_cast<idx>(n));
    for (idx i = 0; i < static_cast<idx>(n); ++i) v[i] = val;
    return v;
}

/// Max absolute error between two CVectors.
static double max_err(const CVector& a, const CVector& b) {
    double e = 0;
    for (idx i = 0; i < a.size(); ++i)
        e = std::max(e, std::abs(a[i] - b[i]));
    return e;
}

static double max_err_real(const Vector& a, const Vector& b) {
    double e = 0;
    for (idx i = 0; i < a.size(); ++i)
        e = std::max(e, std::abs(a[i] - b[i]));
    return e;
}

// Run a single test body against every available backend.
static void for_each_backend(std::function<void(FFTBackend)> body) {
    body(FFTBackend::seq);
    body(FFTBackend::simd);      // falls back to seq on non-SIMD platforms
#ifdef NUMERICS_HAS_STD_SIMD
    body(FFTBackend::stdsimd);
#endif
#ifdef NUMERICS_HAS_FFTW
    body(FFTBackend::fftw);
#endif
}

static std::string backend_name(FFTBackend b) {
    switch (b) {
        case FFTBackend::seq:     return "seq";
        case FFTBackend::simd:    return "simd";
        case FFTBackend::stdsimd: return "stdsimd";
        case FFTBackend::fftw:    return "fftw";
    }
    return "unknown";
}

// ---- DC component -----------------------------------------------------------

TEST(FFT, DCComponent) {
    // x[j] = 1  =>  X[0] = n,  X[k>0] = 0
    for_each_backend([](FFTBackend b) {
        const int n = 64;
        CVector in = make_cvec(n, {1.0, 0.0});
        CVector out = make_cvec(n);
        fft(in, out, b);
        EXPECT_NEAR(out[0].real(), static_cast<double>(n), 1e-10)
            << "backend=" << backend_name(b);
        EXPECT_NEAR(out[0].imag(), 0.0, 1e-10);
        for (int k = 1; k < n; ++k)
            EXPECT_NEAR(std::abs(out[k]), 0.0, 1e-9)
                << "k=" << k;
    });
}

// ---- Single-frequency spike -------------------------------------------------

TEST(FFT, SingleFrequencySpike) {
    // x[j] = exp(2*pi*i * k0*j / n)  =>  X[k0] = n,  all other bins ~0
    for_each_backend([](FFTBackend b) {
        const int n = 64, k0 = 7;
        CVector in = make_cvec(n);
        for (int j = 0; j < n; ++j)
            in[j] = std::exp(cplx{0, TWO_PI * k0 * j / n});
        CVector out = make_cvec(n);
        fft(in, out, b);

        for (int k = 0; k < n; ++k) {
            double expected = (k == k0) ? static_cast<double>(n) : 0.0;
            EXPECT_NEAR(std::abs(out[k]), expected, 1e-8)
                << "k=" << k << " backend=" << backend_name(b);
        }
    });
}

// ---- Round-trip: ifft(fft(x)) / n == x -------------------------------------

TEST(FFT, RoundTrip) {
    for_each_backend([](FFTBackend b) {
        const int n = 128;
        CVector x = make_cvec(n);
        for (int j = 0; j < n; ++j)
            x[j] = cplx{std::sin(TWO_PI * 3 * j / n) + 0.5,
                         std::cos(TWO_PI * 5 * j / n)};
        CVector X = make_cvec(n);
        fft(x, X, b);
        CVector y = make_cvec(n);
        ifft(X, y, b);
        // FFTW convention: ifft is unnormalised -- divide by n
        for (int j = 0; j < n; ++j)
            y[j] /= static_cast<double>(n);

        EXPECT_LT(max_err(x, y), 1e-11)
            << "backend=" << backend_name(b);
    });
}

// ---- Parseval identity ------------------------------------------------------

TEST(FFT, Parseval) {
    // sum_k |X[k]|^2 == n * sum_j |x[j]|^2
    for_each_backend([](FFTBackend b) {
        const int n = 256;
        CVector x = make_cvec(n);
        double energy_x = 0;
        for (int j = 0; j < n; ++j) {
            x[j] = cplx{std::sin(TWO_PI * 11 * j / n),
                         std::cos(TWO_PI * 17 * j / n)};
            energy_x += std::norm(x[j]);
        }
        CVector X = make_cvec(n);
        fft(x, X, b);
        double energy_X = 0;
        for (int k = 0; k < n; ++k)
            energy_X += std::norm(X[k]);

        EXPECT_NEAR(energy_X, static_cast<double>(n) * energy_x, 1e-8)
            << "backend=" << backend_name(b);
    });
}

// ---- Linearity --------------------------------------------------------------

TEST(FFT, Linearity) {
    // fft(a*x + b*y) == a*fft(x) + b*fft(y)
    for_each_backend([](FFTBackend bk) {
        const int n = 64;
        const cplx a{2.0, -1.0}, b{-0.5, 3.0};
        CVector x = make_cvec(n), y = make_cvec(n);
        for (int j = 0; j < n; ++j) {
            x[j] = cplx{std::cos(TWO_PI * 3 * j / n), 0};
            y[j] = cplx{0, std::sin(TWO_PI * 7 * j / n)};
        }
        // Compute a*x + b*y
        CVector xy = make_cvec(n);
        for (int j = 0; j < n; ++j) xy[j] = a * x[j] + b * y[j];

        CVector Fx = make_cvec(n), Fy = make_cvec(n), Fxy = make_cvec(n);
        fft(x, Fx, bk);
        fft(y, Fy, bk);
        fft(xy, Fxy, bk);

        CVector combined = make_cvec(n);
        for (int k = 0; k < n; ++k) combined[k] = a * Fx[k] + b * Fy[k];

        EXPECT_LT(max_err(Fxy, combined), 1e-10)
            << "backend=" << (bk == FFTBackend::seq ? "seq" : "fftw");
    });
}

// ---- rfft: DC and known frequency -------------------------------------------

TEST(FFT, RfftDC) {
    // x[j] = 1  =>  X[0] = n,  X[k>0] = 0
    for_each_backend([](FFTBackend b) {
        const int n = 64;
        Vector x = make_vec(n, 1.0);
        CVector X(static_cast<idx>(n / 2 + 1));
        rfft(x, X, b);
        EXPECT_NEAR(X[0].real(), static_cast<double>(n), 1e-10);
        EXPECT_NEAR(X[0].imag(), 0.0, 1e-10);
        for (int k = 1; k <= n / 2; ++k)
            EXPECT_NEAR(std::abs(X[k]), 0.0, 1e-9) << "k=" << k;
    });
}

TEST(FFT, RfftFrequencySpike) {
    // x[j] = cos(2*pi*k0*j/n)  =>  X[k0].real() = n/2, X[0] = 0
    for_each_backend([](FFTBackend b) {
        const int n = 128, k0 = 5;
        Vector x = make_vec(n);
        for (int j = 0; j < n; ++j)
            x[j] = std::cos(TWO_PI * k0 * j / n);
        CVector X(static_cast<idx>(n / 2 + 1));
        rfft(x, X, b);
        // bin k0 should have magnitude n/2
        EXPECT_NEAR(std::abs(X[k0]), static_cast<double>(n) / 2.0, 1e-8)
            << "backend=" << backend_name(b);
    });
}

// ---- irfft round-trip -------------------------------------------------------

TEST(FFT, IrfftRoundTrip) {
    for_each_backend([](FFTBackend b) {
        const int n = 128;
        Vector x = make_vec(n);
        for (int j = 0; j < n; ++j)
            x[j] = std::sin(TWO_PI * 5 * j / n) + 0.3 * std::cos(TWO_PI * 13 * j / n);
        CVector X(static_cast<idx>(n / 2 + 1));
        rfft(x, X, b);
        Vector y = make_vec(n);
        irfft(X, n, y, b);
        // unnormalised: divide by n
        for (int j = 0; j < n; ++j) y[j] /= static_cast<double>(n);

        EXPECT_LT(max_err_real(x, y), 1e-11)
            << "backend=" << backend_name(b);
    });
}

// ---- FFTPlan ----------------------------------------------------------------

TEST(FFTPlan, MatchesOneShot) {
    for_each_backend([](FFTBackend b) {
        const int n = 256;
        CVector x = make_cvec(n);
        for (int j = 0; j < n; ++j)
            x[j] = cplx{std::cos(TWO_PI * 9 * j / n),
                         std::sin(TWO_PI * 3 * j / n)};
        CVector ref = make_cvec(n);
        fft(x, ref, b);

        FFTPlan plan(n, true, b);
        CVector out = make_cvec(n);
        plan.execute(x, out);

        EXPECT_LT(max_err(ref, out), 1e-12)
            << "backend=" << backend_name(b);
    });
}

TEST(FFTPlan, InversePlanRoundTrip) {
    for_each_backend([](FFTBackend b) {
        const int n = 128;
        CVector x = make_cvec(n);
        for (int j = 0; j < n; ++j)
            x[j] = cplx{static_cast<double>(j % 7), static_cast<double>(j % 5)};

        FFTPlan fwd(n, true,  b);
        FFTPlan inv(n, false, b);

        CVector X = make_cvec(n), y = make_cvec(n);
        fwd.execute(x, X);
        inv.execute(X, y);
        for (int j = 0; j < n; ++j) y[j] /= static_cast<double>(n);

        EXPECT_LT(max_err(x, y), 1e-11)
            << "backend=" << backend_name(b);
    });
}

TEST(FFTPlan, RepeatedExecuteSameResult) {
    // Running the plan multiple times must give the same output each time.
    for_each_backend([](FFTBackend b) {
        const int n = 64;
        CVector x = make_cvec(n);
        for (int j = 0; j < n; ++j) x[j] = cplx{std::cos(j * 0.1), std::sin(j * 0.2)};

        FFTPlan plan(n, true, b);
        CVector out1 = make_cvec(n), out2 = make_cvec(n);
        plan.execute(x, out1);
        plan.execute(x, out2);

        EXPECT_LT(max_err(out1, out2), 1e-15);
    });
}

// ---- All backends must agree with seq --------------------------------------

TEST(FFT, AllBackendsAgree) {
    // Every backend must produce results within floating-point rounding of seq.
    const int n = 512;
    CVector x = make_cvec(n);
    for (int j = 0; j < n; ++j)
        x[j] = cplx{std::sin(TWO_PI * 17 * j / n), std::cos(TWO_PI * 31 * j / n)};

    CVector ref = make_cvec(n);
    fft(x, ref, FFTBackend::seq);

    auto check = [&](FFTBackend b) {
        CVector out = make_cvec(n);
        fft(x, out, b);
        EXPECT_LT(max_err(ref, out), 1e-10) << "backend=" << backend_name(b);
    };
    check(FFTBackend::simd);
#ifdef NUMERICS_HAS_STD_SIMD
    check(FFTBackend::stdsimd);
#endif
#ifdef NUMERICS_HAS_FFTW
    check(FFTBackend::fftw);
#endif
}

TEST(FFT, AllBackendsIrfftAgree) {
    const int n = 256;
    Vector x = make_vec(n);
    for (int j = 0; j < n; ++j) x[j] = std::cos(TWO_PI * 7 * j / n);

    CVector X_ref(static_cast<idx>(n / 2 + 1));
    rfft(x, X_ref, FFTBackend::seq);
    Vector y_ref = make_vec(n);
    irfft(X_ref, n, y_ref, FFTBackend::seq);

    auto check = [&](FFTBackend b) {
        CVector X(static_cast<idx>(n / 2 + 1));
        rfft(x, X, b);
        Vector y = make_vec(n);
        irfft(X, n, y, b);
        EXPECT_LT(max_err_real(y_ref, y), 1e-9) << "backend=" << backend_name(b);
    };
    check(FFTBackend::simd);
#ifdef NUMERICS_HAS_STD_SIMD
    check(FFTBackend::stdsimd);
#endif
#ifdef NUMERICS_HAS_FFTW
    check(FFTBackend::fftw);
#endif
}

// ---- Error handling ---------------------------------------------------------

TEST(FFT, SizeMismatchThrows) {
    CVector in = make_cvec(64);
    CVector out = make_cvec(32);    // wrong size
    EXPECT_THROW(fft(in, out, FFTBackend::seq), std::invalid_argument);
    EXPECT_THROW(ifft(in, out, FFTBackend::seq), std::invalid_argument);
}

TEST(FFT, RfftSizeMismatchThrows) {
    const int n = 64;
    Vector in = make_vec(n);
    CVector out = make_cvec(n);    // should be n/2+1 = 33
    EXPECT_THROW(rfft(in, out, FFTBackend::seq), std::invalid_argument);
}

TEST(FFT, IrfftSizeMismatchThrows) {
    const int n = 64;
    CVector in = make_cvec(n / 2 + 1);
    Vector out = make_vec(n - 1);  // wrong: should be n
    EXPECT_THROW(irfft(in, n, out, FFTBackend::seq), std::invalid_argument);
}

TEST(FFTPlan, ExecuteSizeMismatchThrows) {
    FFTPlan plan(64, true, FFTBackend::seq);
    CVector in = make_cvec(64), out = make_cvec(32);
    EXPECT_THROW(plan.execute(in, out), std::invalid_argument);
}
