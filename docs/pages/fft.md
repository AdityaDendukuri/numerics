# Fast Fourier Transform {#page_fft}

---

## The Discrete Fourier Transform {#sec_dft}

Given a sequence \f$x[0], x[1], \ldots, x[n-1] \in \mathbb{C}\f$, the **Discrete Fourier
Transform** (DFT) maps it to a frequency-domain sequence \f$X[0], \ldots, X[n-1]\f$:

\f[
  X[k] = \sum_{j=0}^{n-1} x[j]\, e^{-2\pi i j k / n}, \qquad k = 0, 1, \ldots, n-1.
\f]

The inverse DFT recovers \f$x\f$ from \f$X\f$:

\f[
  x[j] = \frac{1}{n} \sum_{k=0}^{n-1} X[k]\, e^{+2\pi i j k / n}.
\f]

**Convention note**: both backends follow the FFTW sign convention.
`ifft` returns the *unnormalised* inverse -- divide by \f$n\f$ to recover \f$x\f$ exactly.

### Interpretation of frequency bins

For a signal sampled at rate \f$f_s\f$ (samples per second) with \f$n\f$ samples, bin
\f$k\f$ corresponds to frequency

\f[
  f_k = \frac{k \cdot f_s}{n}.
\f]

Bins \f$k = 0, \ldots, n/2\f$ are non-negative frequencies; bins \f$k = n/2+1, \ldots, n-1\f$
are the negative frequencies (aliased by periodicity). For real input the spectrum is
Hermitian: \f$X[n-k] = \overline{X[k]}\f$, so only the first \f$n/2+1\f$ bins carry
independent information -- this is what `rfft` returns.

### Parseval's theorem

Energy is conserved up to the factor \f$n\f$:

\f[
  \sum_{j=0}^{n-1} |x[j]|^2 = \frac{1}{n} \sum_{k=0}^{n-1} |X[k]|^2.
\f]

---

## Naive DFT: O(n^2) cost {#sec_naive}

Evaluating all \f$n\f$ bins directly costs \f$O(n^2)\f$ multiplications -- impractical
for large \f$n\f$. The key observation that breaks this barrier is the **periodicity and
symmetry** of the twiddle factors \f$W_n^{jk} = e^{-2\pi i jk/n}\f$.

Define the primitive \f$n\f$-th root of unity \f$\omega_n = e^{-2\pi i / n}\f$. Then
\f$W_n^{jk} = \omega_n^{jk}\f$, and these factors satisfy:

\f[
  \omega_{2n}^{2k} = \omega_n^k \quad \text{(halving)}, \qquad
  \omega_n^{k+n/2} = -\omega_n^k \quad \text{(symmetry)}.
\f]

These two identities are the engine of the Cooley-Tukey algorithm.

---

## Cooley-Tukey Radix-2 DIT {#sec_cooley_tukey}

The **Cooley-Tukey radix-2 decimation-in-time** (DIT) algorithm, published in 1965,
reduces an \f$n\f$-point DFT to two \f$n/2\f$-point DFTs by splitting the input into
even- and odd-indexed elements.

### Derivation

Split \f$x\f$ into even and odd halves:

\f[
  X[k] = \sum_{j=0}^{n/2-1} x[2j]\,\omega_n^{2jk}
        + \sum_{j=0}^{n/2-1} x[2j+1]\,\omega_n^{(2j+1)k}.
\f]

Using \f$\omega_n^{2jk} = \omega_{n/2}^{jk}\f$:

\f[
  X[k] = \underbrace{\sum_{j=0}^{n/2-1} x[2j]\,\omega_{n/2}^{jk}}_{E[k]}
        + \omega_n^k \underbrace{\sum_{j=0}^{n/2-1} x[2j+1]\,\omega_{n/2}^{jk}}_{O[k]}.
\f]

where \f$E[k]\f$ and \f$O[k]\f$ are themselves DFTs of length \f$n/2\f$. The symmetry
property \f$\omega_n^{k+n/2} = -\omega_n^k\f$ means the second half of the output costs
nothing extra:

\f[
  X[k]       = E[k] + \omega_n^k\, O[k], \\
  X[k+n/2]   = E[k] - \omega_n^k\, O[k].
\f]

This is the **butterfly operation**: given the pair \f$(u, v)\f$ and twiddle factor
\f$w = \omega_n^k\f$,

\f[
  (u,\; vw) \;\longrightarrow\; (u + vw,\; u - vw).
\f]

Each butterfly costs 1 complex multiply and 2 complex adds. Applying the recursion
\f$\log_2 n\f$ times yields \f$O(n \log n)\f$ total work.

### Recurrence and complexity

Let \f$T(n)\f$ be the operation count. The radix-2 split gives:

\f[
  T(n) = 2\,T(n/2) + O(n), \qquad T(1) = O(1).
\f]

By the Master theorem (case 2): \f$T(n) = O(n \log n)\f$.

For \f$n = 2^{20} \approx 10^6\f$ this is roughly \f$20 \times 10^6\f$ operations versus
\f$10^{12}\f$ for the naive DFT -- a factor of \f$50{,}000\f$ speedup.

### Bit-reversal permutation

The recursive split reorders the input so that element \f$j\f$ ends up at position equal
to the **bit-reversal** of \f$j\f$ in \f$\log_2 n\f$ bits. For example with \f$n = 8\f$:

| Original index | Binary | Bit-reversed | Permuted index |
|---------------|--------|--------------|----------------|
| 0 | 000 | 000 | 0 |
| 1 | 001 | 100 | 4 |
| 2 | 010 | 010 | 2 |
| 3 | 011 | 110 | 6 |
| 4 | 100 | 001 | 1 |
| 5 | 101 | 101 | 5 |
| 6 | 110 | 011 | 3 |
| 7 | 111 | 111 | 7 |

The iterative DIT algorithm applies this permutation in-place first, then processes the
butterfly stages bottom-up (length-2 butterflies, then length-4, ..., then length-\f$n\f$).

---

## Algorithm (iterative DIT) {#sec_algorithm}

```
function FFT(x[0..n-1]):
    bit_reverse_permute(x)
    for len = 2, 4, 8, ..., n:
        w_len = exp(-2*pi*i / len)       // principal twiddle
        for i = 0, len, 2*len, ..., n-len:
            w = 1
            for j = 0 to len/2 - 1:
                u = x[i + j]
                v = x[i + j + len/2] * w
                x[i + j]         = u + v    // butterfly top
                x[i + j + len/2] = u - v    // butterfly bottom
                w *= w_len
```

For the inverse DFT, replace \f$-2\pi\f$ with \f$+2\pi\f$ in the twiddle angle. The
output is unnormalised; divide each element by \f$n\f$ for the true inverse.

**Complexity**: \f$O(n \log n)\f$ multiplications; \f$O(n)\f$ extra memory for the
in-place permutation and butterfly stages.

---

## Real-to-Complex Transform (rfft) {#sec_rfft}

When the input \f$x\f$ is real, \f$X[n-k] = \overline{X[k]}\f$ (Hermitian symmetry), so
only the \f$n/2 + 1\f$ bins \f$k = 0, \ldots, n/2\f$ are independent.

`rfft` returns exactly these \f$n/2 + 1\f$ complex values. `irfft` reconstructs the full
real output from them by:

1. Reconstructing the negative-frequency half via conjugate symmetry:
   \f$X[n-k] = \overline{X[k]}\f$ for \f$k = 1, \ldots, n/2 - 1\f$.
2. Running the full complex inverse DFT on the reconstructed spectrum.
3. Taking the real part (imaginary part is zero to floating-point precision for truly real
   input).

This halves the storage compared to the complex DFT and, in the FFTW backend, uses the
dedicated `fftw_plan_dft_r2c_1d` / `fftw_plan_dft_c2r_1d` plans which exploit the
symmetry for an additional speed improvement.

---

## FFTPlan: amortizing planning cost {#sec_fftplan}

For repeated transforms of the same length, the planning cost (twiddle precomputation or
FFTW measurement) is paid once:

```cpp
num::spectral::FFTPlan plan(1024);          // precomputes twiddles (seq) or
                                            // runs FFTW_MEASURE (fftw)
for (auto& frame : frames)
    plan.execute(frame, spectrum);          // O(n log n), no allocation
```

For the `seq` backend, `FFTPlanImpl` stores all twiddle factors for every butterfly stage,
removing the `cos`/`sin` calls from the hot loop. For the `fftw` backend, `FFTPlanImpl`
wraps an `fftw_plan` and calls `fftw_execute_dft` with the new-array API so the same plan
can be reused across different input buffers.

---

## Backends {#sec_backends}

| Backend | Algorithm | Size constraint | When selected |
|---------|-----------|-----------------|---------------|
| `seq` | Iterative Cooley-Tukey radix-2 DIT | Power of two | Always available |
| `fftw` | FFTW3 (mixed-radix, SIMD) | Any \f$n\f$ | `NUMERICS_HAS_FFTW` defined at configure time |

**Backend selection**: `default_fft_backend` is `fftw` when FFTW3 is found, `seq`
otherwise. An explicit backend can be passed to any function:

```cpp
using namespace num::spectral;
fft(in, out, seq);   // always use native Cooley-Tukey
fft(in, out, fftw);  // always use FFTW3 (throws if not compiled in)
```

### seq backend

Implements the iterative radix-2 DIT described above. Requires \f$n\f$ to be a power of
two; throws `std::invalid_argument` otherwise. Twiddle factors are recomputed on each
call; `FFTPlan` precomputes and caches them.

### fftw backend

Wraps FFTW3 (`libfftw3`). One-shot calls use `FFTW_ESTIMATE` (no measurement). `FFTPlan`
uses `FFTW_MEASURE` -- FFTW benchmarks several internal decompositions at plan construction
time and selects the fastest. The plan is then reused via `fftw_execute_dft` (new-array
API) which is thread-safe for different input/output buffers.

FFTW supports arbitrary \f$n\f$ via mixed-radix decomposition; performance is best for
highly composite \f$n\f$ (especially powers of two, but also products of small primes
such as 2, 3, 5, 7).

---

## API Reference {#sec_api}

```cpp
namespace num::spectral {

// One-shot transforms
void fft  (const CVector& in, CVector& out,  FFTBackend b = default_fft_backend);
void ifft (const CVector& in, CVector& out,  FFTBackend b = default_fft_backend);
void rfft (const Vector&  in, CVector& out,  FFTBackend b = default_fft_backend);
void irfft(const CVector& in, int n, Vector& out, FFTBackend b = default_fft_backend);

// Reusable plan
class FFTPlan {
    explicit FFTPlan(int n, bool forward = true, FFTBackend b = default_fft_backend);
    void execute(const CVector& in, CVector& out) const;
    int        size()    const;
    FFTBackend backend() const;
};

} // namespace num::spectral
```

**Size and allocation requirements**:

| Function | `in.size()` | `out` pre-allocated to |
|----------|------------|------------------------|
| `fft`, `ifft` | any power of two (seq) / any n (fftw) | `in.size()` |
| `rfft` | same constraints | `in.size() / 2 + 1` |
| `irfft` | `n / 2 + 1` | `n` |
| `FFTPlan::execute` | must equal plan size | plan size |

Passing a size mismatch throws `std::invalid_argument`.

---

## Worked Example {#sec_example}

Compute the spectrum of \f$x[j] = \cos(2\pi \cdot 4 \cdot j / 64)\f$, a pure tone at
frequency bin 4 out of 64 samples.

```cpp
#include "spectral/fft.hpp"
#include <cmath>
#include <iostream>

int main() {
    using namespace num::spectral;
    const int n = 64;

    num::Vector  xr(n);
    num::CVector X(n / 2 + 1);
    for (int j = 0; j < n; ++j)
        xr[j] = std::cos(2.0 * M_PI * 4 * j / n);

    rfft(xr, X);   // n/2+1 = 33 complex bins

    // Bin 4 should be non-zero; all others ~0
    for (int k = 0; k < 8; ++k)
        std::cout << "X[" << k << "] = " << std::abs(X[k]) << "\n";
}
```

Expected output (up to floating-point rounding): `X[4] = 32.0`, all others `~0.0`.
The factor of \f$n/2 = 32\f$ arises from the unnormalised convention:
\f$\cos(\theta) = (e^{i\theta} + e^{-i\theta})/2\f$, so bin 4 receives \f$n/2\f$ from
the \f$e^{i\theta}\f$ component.
