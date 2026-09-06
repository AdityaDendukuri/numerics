# Fast Fourier Transforms {#page_spectral}

Complex FFT/IFFT, real-to-complex RFFT/IRFFT, plan-based execution, and Discrete Sine Transforms across pure C++, SIMD, and FFTW3 backends.

---

## 1. Complex Fourier Transform (FFT / IFFT)

Forward transform: \f$X_k = \sum_{n=0}^{N-1} x_n e^{-i 2\pi k n / N}\f$  
Inverse transform: \f$x_n = \frac{1}{N} \sum_{k=0}^{N-1} X_k e^{i 2\pi k n / N}\f$

```cpp
#include <numerics.hpp>

num::cvec x{{1.0, 0.0}, {0.0, 0.0}, {-1.0, 0.0}, {0.0, 0.0}};
num::cvec X(x.size());

num::spectral::fft(x, X); // Forward FFT (unnormalized)

num::cvec x_rec(x.size());
num::spectral::ifft(X, x_rec);
num::scale(x_rec, 1.0 / x.size()); // Normalized inverse (1/N)
```

---

## 2. Real-to-Complex Transform (RFFT / IRFFT)

Exploits Hermitian conjugate symmetry \f$X_{N-k} = X_k^*\f$ to compute only non-negative modes \f$k \in [0, \lfloor N/2 \rfloor]\f$.

```cpp
constexpr int n = 1024;
num::vec x(n, 0.0);
num::cvec X(n / 2 + 1); // Only non-negative frequencies

num::spectral::rfft(x, X);

// Inverse transform
num::vec x_rec(n);
num::spectral::irfft(X, n, x_rec);
num::scale(x_rec, 1.0 / n);
```

---

## 3. Plan-Based Execution (num::spectral::fft_plan)

Precomputes twiddles and bit-reversal tables for repeated transforms of length \f$N\f$:

```cpp
num::spectral::fft_plan plan(n);
num::cvec in(n), out(n);

plan.execute(in, out); // Reuses twiddle factors; zero allocations
```

---

## 4. Discrete Sine Transform (DST-I)

Diagonalizes the 1D and 2D Dirichlet Laplacian. Size \f$N\f$ must equal \f$2^p - 1\f$ (3, 7, 15, 31, 63, ...).

\f[
X_k = \sum_{j=1}^{N} x_j \sin\left(\frac{j k \pi}{N+1}\right), \qquad k = 1, \dots, N
\f]

```cpp
// 1D DST-I
num::vec x{1.0, 2.0, 3.0};
num::vec X = num::spectral::dst1(x);

// 2D DST-I in place on an N x N row-major grid
constexpr int N = 7;
std::vector<double> grid(N * N, 1.0);
num::spectral::dst2d(grid, N);
```

---

## 5. Backends

| Backend Tag | Description | Requirements |
| :--- | :--- | :--- |
| `fft_backend::seq` | scalar Cooley–Tukey Radix-2 | Standard C++ |
| `fft_backend::simd` | Explicit AVX2 / ARM NEON vectorization | Target CPU flags |
| `fft_backend::fftw` | Vendor FFTW3 library | `NUMERICS_HAS_FFTW` |

```cpp
num::spectral::fft(in, out, num::spectral::default_fft_backend);
```

---

## Complete Example

@example 07_spectral_fft_transforms.cpp

