/// @file 07_spectral_fft_transforms.cpp
/// @brief 1D/2D FFT, iFFT, Real rFFT/irfft, and Precomputed FFTPlan.
#include <cmath>
#include <iostream>
#include <numerics.hpp>

int main() {
    using namespace num;
    using namespace num::spectral;

    idx n = 64;
    Vector r_in(n, 0.0);
    std::vector<double> t_grid(n);
    for (idx i = 0; i < n; ++i) {
        t_grid[i] = (2.0 * M_PI / n) * i;
        r_in[i] = std::sin(3.0 * t_grid[i]) + (0.5 * std::cos(7.0 * t_grid[i]));
    }

    CVector r_out((n / 2) + 1, cplx{0, 0});
    rfft(r_in, r_out, FFTBackend::seq);

    std::vector<double> k_freq((n / 2) + 1), mag_spec((n / 2) + 1);
    for (idx k = 0; k < (n / 2) + 1; ++k) {
        k_freq[k] = static_cast<double>(k);
        mag_spec[k] = std::abs(r_out[k]);
    }

    std::cout << "rFFT Real Signal transformed into " << mag_spec.size() << " frequency bins.\n";

    plt::plot(t_grid, r_in, "Signal r_in(t)", "lines");
    plt::title("07 Spectral FFT: Input Dual-Frequency Signal");
    plt::xlabel("Time t");
    plt::ylabel("Amplitude");
    plt::show_dumb(140, 30);

    plt::plot(k_freq, mag_spec, "Spectrum |X(k)|", "linespoints");
    plt::title("07 Spectral FFT: Frequency Spectrum Magnitude");
    plt::xlabel("Frequency Bin k");
    plt::ylabel("|X(k)|");
    plt::show_dumb(140, 30);

    return 0;
}
