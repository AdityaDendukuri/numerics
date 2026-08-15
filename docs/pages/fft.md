# FFT Examples {#page_fft}

`include/spectral/fft.hpp` provides complex FFT, inverse FFT, real-input FFT,
inverse real FFT, and reusable plans.

## Real Signal Spectrum

```cpp
#include <numerics.hpp>
#include <cmath>
#include <numbers>

constexpr int N = 1024;
constexpr double fs = 1000.0;
constexpr double pi = std::numbers::pi;

num::Vector x(N);
for (int i = 0; i < N; ++i) {
    double t = static_cast<double>(i) / fs;
    x[i] = std::sin(2.0 * pi * 50.0 * t)
         + 0.5 * std::sin(2.0 * pi * 120.0 * t);
}

num::CVector X(N / 2 + 1);
num::spectral::rfft(x, X);
```

Frequency bin \f$k\f$ corresponds to

\f[
    f_k = k f_s / N .
\f]

```cpp
num::Series spectrum;
for (num::idx k = 0; k < X.size(); ++k) {
    spectrum.store(static_cast<double>(k) * fs / N, std::abs(X[k]));
}
```

## Complex Round Trip

```cpp
num::CVector x(N), X(N), y(N);

num::spectral::fft(x, X);
num::spectral::ifft(X, y);
```

The inverse routine returns the normalized inverse transform.

## Reusable Plan

```cpp
num::spectral::FFTPlan plan(N, true, num::spectral::default_fft_backend);

for (int sample = 0; sample < nsamples; ++sample) {
    load_sample(sample, x);
    plan.execute(x, X);
}
```

## Explicit Backend

```cpp
num::spectral::fft(x, X, num::spectral::FFTBackend::seq);
num::spectral::fft(x, X, num::spectral::FFTBackend::simd);
num::spectral::fft(x, X, num::spectral::FFTBackend::stdsimd);
num::spectral::fft(x, X, num::spectral::FFTBackend::fftw);
```

`default_fft_backend` selects FFTW3 when available, otherwise the best built-in
backend.
