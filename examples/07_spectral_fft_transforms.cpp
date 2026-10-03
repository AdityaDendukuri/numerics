/// @file 07_spectral_fft_transforms.cpp
/// @brief FFT and inverse FFT, the real FFT, and the backend that runs them.
///
/// `fft` and `ifft` transform complex vectors, and `rfft` and `irfft` transform real ones, keeping
/// the n/2 + 1 non-negative frequencies. The inverse transforms are unnormalized, as in FFTW, so
/// dividing by n recovers the input. The caller sizes the output vectors. The backend defaults to
/// FFTW when it is present, then the SIMD path, then the portable one, and each call can name one
/// explicitly.
#include <cmath>
#include <cstdio>
#include <numbers>
#include <numerics.hpp>
#include <string_view>
#include <vector>

using namespace num;
using namespace num::spectral;

int main(int argc, char **argv) {
    const bool plot = argc > 1 && std::string_view(argv[1]) == "--plot";
    std::printf("FFTW available %d, SIMD path available %d\n", has_fftw, has_fft_simd);

    // A signal with components at frequencies 3 and 7.
    constexpr idx n = 64;
    vec<real> signal(n);
    for (idx i = 0; i < n; ++i) {
        const real t = 2.0 * std::numbers::pi * static_cast<real>(i) / n;
        signal[i] = std::sin(3.0 * t) + (0.5 * std::cos(7.0 * t));
    }

    vec<cplx> spectrum((n / 2) + 1); // rfft writes the n/2 + 1 non-negative frequencies
    rfft(signal, spectrum);
    std::printf("rfft: %zu bins; amplitudes 2|X(k)|/n above 0.01:\n",
                static_cast<std::size_t>(spectrum.size()));
    for (idx k = 0; k < spectrum.size(); ++k) {
        const real amplitude = 2.0 * std::abs(spectrum[k]) / n;
        if (amplitude > 0.01) {
            std::printf("  k = %2zu  amplitude %.4f\n", static_cast<std::size_t>(k), amplitude);
        }
    }

    vec<real> back(n);
    irfft(spectrum, static_cast<int>(n), back);
    real error = 0.0;
    for (idx i = 0; i < n; ++i) {
        error = std::max(error, std::abs((back[i] / n) - signal[i]));
    }
    std::printf("irfft(rfft(x)) / n - x: %.1e\n", error);

    // Complex round trip, through the portable backend by name.
    vec<cplx> z(n), Z(n), z_back(n);
    for (idx i = 0; i < n; ++i) {
        z[i] = cplx(std::cos(0.3 * i), std::sin(0.1 * i));
    }
    fft(z, Z, fft_backend::seq);
    ifft(Z, z_back, fft_backend::seq);
    error = 0.0;
    for (idx i = 0; i < n; ++i) {
        error = std::max(error, std::abs((z_back[i] / static_cast<real>(n)) - z[i]));
    }
    std::printf("ifft(fft(z)) / n - z:   %.1e\n", error);

    if (plot) {
        std::vector<double> k, magnitude;
        for (idx i = 0; i < spectrum.size(); ++i) {
            k.push_back(static_cast<double>(i));
            magnitude.push_back(std::abs(spectrum[i]));
        }
        plt::plot(k, magnitude, "|X(k)|", "linespoints");
        plt::title("07 Spectrum");
        plt::show();
    }
}
