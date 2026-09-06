/// @file test_fft.cpp
/// @brief Unit tests for num::spectral FFT module.

#include "core/types.hpp"
#include "container/vector_ops.hpp"
#include "spectral/fft.hpp"
#include <cmath>
#include <complex>
#include <functional>
#include <gtest/gtest.h>
#include <stdexcept>

using namespace num;
using namespace num::spectral;
using cplx = std::complex<double>;

static constexpr double TWO_PI = 6.283185307179586476925286766559;

// Helpers

static cvec make_cvec(int n, cplx val = cplx{}) {
    cvec v(static_cast<idx>(n));
    for (idx i = 0; i < static_cast<idx>(n); ++i) {
        v[i] = val;
    }
    return v;
}

static vec make_vec(int n, real val = 0.0) {
    vec v(static_cast<idx>(n));
    for (idx i = 0; i < static_cast<idx>(n); ++i) {
        v[i] = val;
    }
    return v;
}

static double max_err(const cvec &a, const cvec &b) {
    double e = 0;
    for (idx i = 0; i < a.size(); ++i) {
        e = std::max(e, std::abs(a[i] - b[i]));
    }
    return e;
}

static double max_err_real(const vec &a, const vec &b) {
    double e = 0;
    for (idx i = 0; i < a.size(); ++i) {
        e = std::max(e, std::abs(a[i] - b[i]));
    }
    return e;
}

static void for_each_backend(std::function<void(fft_backend)> body) {
    body(fft_backend::seq);
    body(fft_backend::simd); // falls back to seq on non-SIMD platforms
#ifdef NUMERICS_HAS_STD_SIMD
    body(fft_backend::stdsimd);
#endif
#ifdef NUMERICS_HAS_FFTW
    body(fft_backend::fftw);
#endif
}

static std::string backend_name(fft_backend b) {
    switch (b) {
    case fft_backend::seq:
        return "seq";
    case fft_backend::simd:
        return "simd";
    case fft_backend::stdsimd:
        return "stdsimd";
    case fft_backend::fftw:
        return "fftw";
    }
    return "unknown";
}

// DC component

TEST(FFT, DCComponent) {
    // Constant input has only the zero-frequency bin.
    for_each_backend([](fft_backend b) {
        const int n = 64;
        cvec in = make_cvec(n, {1.0, 0.0});
        cvec out = make_cvec(n);
        fft(in, out, b);
        EXPECT_NEAR(out[0].real(), static_cast<double>(n), 1e-10) << "backend=" << backend_name(b);
        EXPECT_NEAR(out[0].imag(), 0.0, 1e-10);
        for (int k = 1; k < n; ++k) {
            EXPECT_NEAR(std::abs(out[k]), 0.0, 1e-9) << "k=" << k;
        }
    });
}

// Single-frequency spike

TEST(FFT, SingleFrequencySpike) {
    // A complex sinusoid maps to one Fourier bin.
    for_each_backend([](fft_backend b) {
        const int n = 64, k0 = 7;
        cvec in = make_cvec(n);
        for (int j = 0; j < n; ++j) {
            in[j] = std::exp(cplx{0, TWO_PI * k0 * j / n});
        }
        cvec out = make_cvec(n);
        fft(in, out, b);

        for (int k = 0; k < n; ++k) {
            double expected = (k == k0) ? static_cast<double>(n) : 0.0;
            EXPECT_NEAR(std::abs(out[k]), expected, 1e-8)
                << "k=" << k << " backend=" << backend_name(b);
        }
    });
}

// Round-trip

TEST(FFT, RoundTrip) {
    for_each_backend([](fft_backend b) {
        const int n = 128;
        cvec x = make_cvec(n);
        for (int j = 0; j < n; ++j) {
            x[j] = cplx{std::sin(TWO_PI * 3 * j / n) + 0.5, std::cos(TWO_PI * 5 * j / n)};
        }
        cvec X = make_cvec(n);
        fft(x, X, b);
        cvec y = make_cvec(n);
        ifft(X, y, b);
        // Backends use an unnormalized inverse.
        for (int j = 0; j < n; ++j) {
            y[j] /= static_cast<double>(n);
        }

        EXPECT_LT(max_err(x, y), 1e-11) << "backend=" << backend_name(b);
    });
}

// Parseval identity

TEST(FFT, Parseval) {
    for_each_backend([](fft_backend b) {
        const int n = 256;
        cvec x = make_cvec(n);
        double energy_x = 0;
        for (int j = 0; j < n; ++j) {
            x[j] = cplx{std::sin(TWO_PI * 11 * j / n), std::cos(TWO_PI * 17 * j / n)};
            energy_x += std::norm(x[j]);
        }
        cvec X = make_cvec(n);
        fft(x, X, b);
        double energy_X = 0;
        for (int k = 0; k < n; ++k) {
            energy_X += std::norm(X[k]);
        }

        EXPECT_NEAR(energy_X, static_cast<double>(n) * energy_x, 1e-8)
            << "backend=" << backend_name(b);
    });
}

// Linearity

TEST(FFT, Linearity) {
    for_each_backend([](fft_backend bk) {
        const int n = 64;
        const cplx a{2.0, -1.0}, b{-0.5, 3.0};
        cvec x = make_cvec(n), y = make_cvec(n);
        for (int j = 0; j < n; ++j) {
            x[j] = cplx{std::cos(TWO_PI * 3 * j / n), 0};
            y[j] = cplx{0, std::sin(TWO_PI * 7 * j / n)};
        }
        cvec xy = make_cvec(n);
        for (int j = 0; j < n; ++j) {
            xy[j] = a * x[j] + b * y[j];
        }

        cvec Fx = make_cvec(n), Fy = make_cvec(n), Fxy = make_cvec(n);
        fft(x, Fx, bk);
        fft(y, Fy, bk);
        fft(xy, Fxy, bk);

        cvec combined = make_cvec(n);
        for (int k = 0; k < n; ++k) {
            combined[k] = a * Fx[k] + b * Fy[k];
        }

        EXPECT_LT(max_err(Fxy, combined), 1e-10)
            << "backend=" << (bk == fft_backend::seq ? "seq" : "fftw");
    });
}

// Real-input FFT

TEST(FFT, RfftDC) {
    // Constant input has only the zero-frequency bin.
    for_each_backend([](fft_backend b) {
        const int n = 64;
        vec x = make_vec(n, 1.0);
        cvec X(static_cast<idx>((n / 2) + 1));
        rfft(x, X, b);
        EXPECT_NEAR(X[0].real(), static_cast<double>(n), 1e-10);
        EXPECT_NEAR(X[0].imag(), 0.0, 1e-10);
        for (int k = 1; k <= n / 2; ++k) {
            EXPECT_NEAR(std::abs(X[k]), 0.0, 1e-9) << "k=" << k;
        }
    });
}

TEST(FFT, RfftFrequencySpike) {
    for_each_backend([](fft_backend b) {
        const int n = 128, k0 = 5;
        vec x = make_vec(n);
        for (int j = 0; j < n; ++j) {
            x[j] = std::cos(TWO_PI * k0 * j / n);
        }
        cvec X(static_cast<idx>((n / 2) + 1));
        rfft(x, X, b);
        EXPECT_NEAR(std::abs(X[k0]), static_cast<double>(n) / 2.0, 1e-8)
            << "backend=" << backend_name(b);
    });
}

// Real-input inverse round-trip

TEST(FFT, IrfftRoundTrip) {
    for_each_backend([](fft_backend b) {
        const int n = 128;
        vec x = make_vec(n);
        for (int j = 0; j < n; ++j) {
            x[j] = std::sin(TWO_PI * 5 * j / n) + (0.3 * std::cos(TWO_PI * 13 * j / n));
        }
        cvec X(static_cast<idx>((n / 2) + 1));
        rfft(x, X, b);
        vec y = make_vec(n);
        irfft(X, n, y, b);
        // Backends use an unnormalized inverse.
        for (int j = 0; j < n; ++j) {
            y[j] /= static_cast<double>(n);
        }

        EXPECT_LT(max_err_real(x, y), 1e-11) << "backend=" << backend_name(b);
    });
}

// fft_plan

TEST(fft_plan, MatchesOneShot) {
    for_each_backend([](fft_backend b) {
        const int n = 256;
        cvec x = make_cvec(n);
        for (int j = 0; j < n; ++j) {
            x[j] = cplx{std::cos(TWO_PI * 9 * j / n), std::sin(TWO_PI * 3 * j / n)};
        }
        cvec ref = make_cvec(n);
        fft(x, ref, b);

        fft_plan plan(n, true, b);
        cvec out = make_cvec(n);
        plan.execute(x, out);

        EXPECT_LT(max_err(ref, out), 1e-12) << "backend=" << backend_name(b);
    });
}

TEST(fft_plan, InversePlanRoundTrip) {
    for_each_backend([](fft_backend b) {
        const int n = 128;
        cvec x = make_cvec(n);
        for (int j = 0; j < n; ++j) {
            x[j] = cplx{static_cast<double>(j % 7), static_cast<double>(j % 5)};
        }

        fft_plan fwd(n, true, b);
        fft_plan inv(n, false, b);

        cvec X = make_cvec(n), y = make_cvec(n);
        fwd.execute(x, X);
        inv.execute(X, y);
        for (int j = 0; j < n; ++j) {
            y[j] /= static_cast<double>(n);
        }

        EXPECT_LT(max_err(x, y), 1e-11) << "backend=" << backend_name(b);
    });
}

TEST(fft_plan, RepeatedExecuteSameResult) {
    for_each_backend([](fft_backend b) {
        const int n = 64;
        cvec x = make_cvec(n);
        for (int j = 0; j < n; ++j) {
            x[j] = cplx{std::cos(j * 0.1), std::sin(j * 0.2)};
        }

        fft_plan plan(n, true, b);
        cvec out1 = make_cvec(n), out2 = make_cvec(n);
        plan.execute(x, out1);
        plan.execute(x, out2);

        EXPECT_LT(max_err(out1, out2), 1e-15);
    });
}

// Backend consistency

TEST(FFT, AllBackendsAgree) {
    const int n = 512;
    cvec x = make_cvec(n);
    for (int j = 0; j < n; ++j) {
        x[j] = cplx{std::sin(TWO_PI * 17 * j / n), std::cos(TWO_PI * 31 * j / n)};
    }

    cvec ref = make_cvec(n);
    fft(x, ref, fft_backend::seq);

    auto check = [&](fft_backend b) {
        cvec out = make_cvec(n);
        fft(x, out, b);
        EXPECT_LT(max_err(ref, out), 1e-10) << "backend=" << backend_name(b);
    };
    check(fft_backend::simd);
#ifdef NUMERICS_HAS_STD_SIMD
    check(fft_backend::stdsimd);
#endif
#ifdef NUMERICS_HAS_FFTW
    check(fft_backend::fftw);
#endif
}

TEST(FFT, AllBackendsIrfftAgree) {
    const int n = 256;
    vec x = make_vec(n);
    for (int j = 0; j < n; ++j) {
        x[j] = std::cos(TWO_PI * 7 * j / n);
    }

    cvec X_ref(static_cast<idx>((n / 2) + 1));
    rfft(x, X_ref, fft_backend::seq);
    vec y_ref = make_vec(n);
    irfft(X_ref, n, y_ref, fft_backend::seq);

    auto check = [&](fft_backend b) {
        cvec X(static_cast<idx>((n / 2) + 1));
        rfft(x, X, b);
        vec y = make_vec(n);
        irfft(X, n, y, b);
        EXPECT_LT(max_err_real(y_ref, y), 1e-9) << "backend=" << backend_name(b);
    };
    check(fft_backend::simd);
#ifdef NUMERICS_HAS_STD_SIMD
    check(fft_backend::stdsimd);
#endif
#ifdef NUMERICS_HAS_FFTW
    check(fft_backend::fftw);
#endif
}

// Error handling

TEST(FFT, SizeMismatchThrows) {
    cvec in = make_cvec(64);
    cvec out = make_cvec(32); // wrong size
    EXPECT_THROW(fft(in, out, fft_backend::seq), std::invalid_argument);
    EXPECT_THROW(ifft(in, out, fft_backend::seq), std::invalid_argument);
}

TEST(FFT, RfftSizeMismatchThrows) {
    const int n = 64;
    vec in = make_vec(n);
    cvec out = make_cvec(n); // should be n/2+1 = 33
    EXPECT_THROW(rfft(in, out, fft_backend::seq), std::invalid_argument);
}

TEST(FFT, IrfftSizeMismatchThrows) {
    const int n = 64;
    cvec in = make_cvec((n / 2) + 1);
    vec out = make_vec(n - 1); // wrong: should be n
    EXPECT_THROW(irfft(in, n, out, fft_backend::seq), std::invalid_argument);
}

TEST(fft_plan, ExecuteSizeMismatchThrows) {
    fft_plan plan(64, true, fft_backend::seq);
    cvec in = make_cvec(64), out = make_cvec(32);
    EXPECT_THROW(plan.execute(in, out), std::invalid_argument);
}
