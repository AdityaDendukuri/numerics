/// @file 07_spectral_fft_transforms.cpp
/// @brief FFT and inverse FFT, the real FFT, and the backend that runs them.
///
/// `fft` and `ifft` transform complex vectors, and `rfft` and `irfft` transform real ones, keeping
/// the n/2 + 1 non-negative frequencies. The inverse transforms are unnormalized, so dividing by n
/// recovers the input. The caller sizes the output vectors. The backend defaults to the SIMD path
/// when the target has one, else the portable one, and each call can name one explicitly.
#include <cmath>
#include <io/json.hpp>
#include <numbers>
#include <numerics.hpp>
#include <string_view>
#include <vector>

using namespace num;
using namespace num::spectral;

int main(int argc, char **argv) {
    const bool plot = argc > 1 && std::string_view(argv[1]) == "--plot";
    io::json results = {{"simd_available", has_fft_simd}};

    // A signal with components at frequencies 3 and 7.
    constexpr idx n = 64;
    vec<real> signal(n);
    for (idx i = 0; i < n; ++i) {
        const real t = 2.0 * std::numbers::pi * static_cast<real>(i) / n;
        signal[i] = std::sin(3.0 * t) + (0.5 * std::cos(7.0 * t));
    }

    vec<cplx> spectrum((n / 2) + 1); // rfft writes the n/2 + 1 non-negative frequencies
    rfft(signal, spectrum);
    results["frequency_bins"] = spectrum.size();
    results["peaks"] = io::json::array();
    for (idx k = 0; k < spectrum.size(); ++k) {
        const real amplitude = 2.0 * std::abs(spectrum[k]) / n;
        if (amplitude > 0.01) {
            results["peaks"].push_back({{"frequency", k}, {"amplitude", amplitude}});
        }
    }

    vec<real> back(n);
    irfft(spectrum, static_cast<int>(n), back);
    real error = 0.0;
    for (idx i = 0; i < n; ++i) {
        error = std::max(error, std::abs((back[i] / n) - signal[i]));
    }
    results["real_round_trip_error"] = error;

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
    results["complex_round_trip_error"] = error;
    io::write_json(results, "result.json");

    if (plot) {
        std::vector<double> k, magnitude;
        for (idx i = 0; i < spectrum.size(); ++i) {
            k.push_back(static_cast<double>(i));
            magnitude.push_back(std::abs(spectrum[i]));
        }
        plt::plot(k, magnitude, "|X(k)|", "linespoints");
        plt::title("07 Spectrum");
        plt::savefig("plot.png");
    }
}
