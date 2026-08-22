# FFT Examples {#page_fft}

## Complex Forward Transform

```cpp
#include <numerics.hpp>

num::CVector signal{{1.0, 0.0}, {0.0, 0.0}, {-1.0, 0.0}, {0.0, 0.0}};
num::CVector spectrum(signal.size());

num::spectral::fft(signal, spectrum); // Forward transform is unnormalized.
```

## Complex Inverse Transform

```cpp
num::CVector reconstructed(signal.size());
num::spectral::ifft(spectrum, reconstructed); // Inverse transform is also unnormalized.

num::scale(1.0 / signal.size(), reconstructed); // Recover the original normalization.
```

## Real Forward Transform

```cpp
constexpr int n = 1024;
num::Vector signal(n, 0.0);
num::CVector spectrum(n / 2 + 1); // Real input stores only nonnegative frequencies.

num::spectral::rfft(signal, spectrum);
```

## Real Inverse Transform

```cpp
num::Vector reconstructed(n);
num::spectral::irfft(spectrum, n, reconstructed); // Rebuild the conjugate half internally.

num::scale(1.0 / n, reconstructed); // Apply inverse normalization.
```

## Frequency Coordinates

```cpp
constexpr double sample_rate = 1000.0;
num::idx k = 50;
double frequency = static_cast<double>(k) * sample_rate / n; // Frequency of bin k.
double amplitude = std::abs(spectrum[k]);                    // Magnitude at that bin.
```

## Reusable Forward Plan

```cpp
num::spectral::FFTPlan plan(n); // Precompute a forward plan.
num::CVector input(n), output(n);

plan.execute(input, output); // Reuse plan-owned twiddle data.
```

## Reusable Inverse Plan

```cpp
num::spectral::FFTPlan inverse_plan(
    n, false, num::spectral::default_fft_backend); // false selects the inverse transform.

inverse_plan.execute(output, input);
num::scale(1.0 / n, input);
```

## Inspecting a Plan

```cpp
int transform_size = plan.size();                  // Planned vector length.
num::spectral::FFTBackend backend = plan.backend(); // Selected implementation.
```

`FFTPlan` owns its precomputation, can be moved, and cannot be copied.

## Backend Selection

```cpp
num::spectral::fft(input, output, num::spectral::FFTBackend::seq); // Scalar radix-2 path.
```

```cpp
if constexpr (num::spectral::has_fft_simd) {
    num::spectral::fft(input, output, num::spectral::FFTBackend::simd);
}
```

```cpp
if constexpr (num::spectral::has_fft_stdsimd) {
    num::spectral::fft(input, output, num::spectral::FFTBackend::stdsimd);
}
```

```cpp
if constexpr (num::spectral::has_fftw) {
    num::spectral::fft(input, output, num::spectral::FFTBackend::fftw);
}
```

`default_fft_backend` selects the configured default implementation.

## Size Errors

```cpp
num::CVector wrong_size(n - 1);
num::spectral::fft(input, wrong_size); // Throws: input and output sizes differ.
```

The built-in transforms require a power-of-two length.

## Complete Program

@example 07_spectral_fft_transforms.cpp
