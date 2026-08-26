# Fast Fourier Transforms {#page_spectral}

The `spectral` module provides Fast Fourier Transforms (FFT), Real-to-Complex transforms (RFFT), and plan-based spectral acceleration across pure C++ radix-2, SIMD vectorization, and FFTW3 backends.

---

## 1. Complex-to-Complex Discrete Fourier Transform (DFT)

The forward Discrete Fourier Transform maps a discrete signal \f$x \in \mathbb{C}^N\f$ to its frequency spectrum \f$X \in \mathbb{C}^N\f$:

\f[
X_k = \sum_{n=0}^{N-1} x_n \exp\left(-i \frac{2\pi k n}{N}\right), \qquad k = 0, \dots, N-1
\f]

The Inverse Discrete Fourier Transform (IDFT) reconstructs the original signal:

\f[
x_n = \frac{1}{N} \sum_{k=0}^{N-1} X_k \exp\left(i \frac{2\pi k n}{N}\right), \qquad n = 0, \dots, N-1
\f]

```cpp
#include <numerics.hpp>

num::CVector signal{{1.0, 0.0}, {0.0, 0.0}, {-1.0, 0.0}, {0.0, 0.0}};
num::CVector spectrum(signal.size());

// Forward FFT (unnormalized)
num::spectral::fft(signal, spectrum);

// Inverse FFT
num::CVector reconstructed(signal.size());
num::spectral::ifft(spectrum, reconstructed);
num::scale(1.0 / signal.size(), reconstructed); // Normalization scale 1/N
```

---

## 2. Real-to-Complex Fourier Transform (RFFT)

When input data \f$x_n \in \mathbb{R}\f$ is purely real, the Fourier coefficients satisfy Hermitian conjugate symmetry:

\f[
X_{N-k} = X_k^*
\f]

`num::spectral::rfft` exploits this symmetry to compute only the \f$\lfloor N/2 \rfloor + 1\f$ non-negative frequency modes, cutting computational work and memory by half.

\f[
X_k = \sum_{n=0}^{N-1} x_n \exp\left(-i \frac{2\pi k n}{N}\right), \qquad k = 0, 1, \dots, \frac{N}{2}
\f]

```cpp
constexpr int n = 1024;
num::Vector signal(n, 0.0);
num::CVector spectrum(n / 2 + 1); // Only non-negative frequencies k in [0, N/2]

num::spectral::rfft(signal, spectrum);

// Inverse RFFT reconstruction
num::Vector reconstructed(n);
num::spectral::irfft(spectrum, n, reconstructed);
num::scale(1.0 / n, reconstructed);
```

### Frequency Coordinates & Physical Spectral Power

For sampling frequency \f$f_s\f$, each frequency bin \f$k\f$ maps to physical frequency:

\f[
f_k = \frac{k \cdot f_s}{N}, \qquad P_k = |X_k|^2
\f]

```cpp
constexpr double sample_rate = 1000.0; // fs = 1000 Hz
num::idx k = 50;
double frequency = static_cast<double>(k) * sample_rate / n; // Frequency in Hz
double amplitude = std::abs(spectrum[k]);                    // Spectral magnitude |X_k|
```

---

## 3. Reusable Plan-Based Execution (FFTPlan)

For repeated transforms of identical size \f$N\f$, precomputing twiddle factors \f$W_N^k = \exp(-i 2\pi k / N)\f$ and bit-reversal tables eliminates initialization overhead:

```cpp
// Precompute forward plan
num::spectral::FFTPlan plan(n);
num::CVector input(n), output(n);

plan.execute(input, output); // Zero allocation, reuses precomputed twiddles
```

---

## 4. Hardware Backend Selection

| Backend | Implementation | Requirement |
| :--- | :--- | :--- |
| `FFTBackend::seq` | Scalar Cooley–Tukey Radix-2 | Standard C++ |
| `FFTBackend::simd` | Explicit AVX2 / ARM NEON vectorization | Target architecture flags |
| `FFTBackend::fftw` | FFTW3 vendor-optimized engine | `NUMERICS_HAS_FFTW` |

```cpp
num::spectral::fft(input, output, num::spectral::default_fft_backend);
```

---

## 5. Discrete Sine Transform (DST-I)

The DST-I diagonalizes the Dirichlet Laplacian, which is what turns a direct
Poisson solve on a uniform grid into an \f$\mathcal{O}(N^2 \log N)\f$ operation.
It is computed through the complex FFT by odd extension, so it runs on whichever
backend the build selected.

\f[
X_k = \sum_{j=1}^{N} x_j \sin\!\left(\frac{jk\pi}{N+1}\right), \qquad k = 1,\dots,N
\f]

The odd extension has length \f$2(N+1)\f$ and the radix-2 FFT needs that to be a
power of two, so \f$N\f$ must equal \f$2^p - 1\f$ (7, 15, 31, 63, and so on).

```cpp
num::Vector x{1.0, 2.0, 3.0}; // N = 3.
num::Vector X = num::spectral::dst1(x); // Unnormalized DST-I.
```

Applying it twice scales by \f$(N+1)/2\f$, which is how the transform is inverted.

The two-dimensional transform runs the one-dimensional transform along the
columns and then the rows of a row-major \f$N \times N\f$ grid, in place:

```cpp
const int N = 7;
std::vector<double> grid(N * N);
num::spectral::dst2d(grid, N);
```

---

## Complete Example

@example 07_spectral_fft_transforms.cpp
