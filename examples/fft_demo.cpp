/// @file examples/fft_demo.cpp
/// @brief Power spectrum of a composite signal via num::spectral::rfft.
///
/// Signal: two sinusoids at 50 Hz and 120 Hz sampled at 1 kHz.
/// Top panel: time-domain signal (first 100 ms).
/// Bottom panel: |X[k]|, with peaks at both constituent frequencies.

#include "plot/plot.hpp"
#include "spectral/fft.hpp"
#include <cmath>

int main() {
    const int    N  = 1024;
    const double fs = 1000.0;   // sample rate (Hz)

    num::Vector x(N);
    for (int i = 0; i < N; ++i)
        x[i] = std::sin(2 * M_PI * 50.0  * i / fs)
             + 0.5 * std::sin(2 * M_PI * 120.0 * i / fs);

    num::CVector X(N / 2 + 1);
    num::spectral::rfft(x, X);

    num::Series signal, spectrum;
    for (int i = 0; i < 100; ++i)
        signal.store(i / fs * 1000.0, x[i]);   // first 100 ms
    for (int k = 0; k < (int)X.size(); ++k)
        spectrum.store(k * fs / N, std::abs(X[k]));

    num::plt::subplot(2, 1);
    num::plt::plot(signal);
    num::plt::title("Signal: 50 Hz + 0.5 * 120 Hz");
    num::plt::xlabel("time (ms)");
    num::plt::ylabel("amplitude");

    num::plt::next();
    num::plt::plot(spectrum);
    num::plt::title("FFT power spectrum");
    num::plt::xlabel("frequency (Hz)");
    num::plt::ylabel("|X[k]|");

    num::plt::savefig("fft_demo.png");
}
