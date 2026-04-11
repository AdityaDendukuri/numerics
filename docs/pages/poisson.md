# 2D Poisson Solver {#page_poisson}

**Header:** `#include "pde/poisson.hpp"`  
**Namespace:** `num::pde`

Solves \f$-\nabla^2 u = f\f$ on \f$(0,1)^2\f$ with homogeneous Dirichlet boundary
conditions, discretised on an \f$N \times N\f$ interior grid with mesh spacing
\f$h = 1/(N+1)\f$.  Two variants are provided: a finite-difference solver with
\f$O(h^2)\f$ error and a spectral solver with machine-precision error for \f$f\f$
in the DST eigenbasis.  Both run in \f$O(N^2 \log N)\f$ via the Discrete Sine
Transform.

Reference: J. Demmel, *Parallel Spectral Methods: Fast Fourier Transform with
Applications*, CS267 Lecture 20, UC Berkeley, 2025.

---

## Problem formulation {#sec_poisson_problem}

Find \f$u : (0,1)^2 \to \mathbb{R}\f$ such that

\f[
  -\nabla^2 u = f \quad \text{on } (0,1)^2, \qquad u = 0 \text{ on } \partial\Omega.
\f]

Discretise on the \f$N \times N\f$ interior grid \f$(x_i, y_j) = (ih, jh)\f$,
\f$i,j = 1,\ldots,N\f$, using the standard 5-point Laplacian.  The discrete system is

\f[
  L_2\, \mathbf{u} = h^2\, \mathbf{f},
\f]

where \f$L_2 = L_1 \otimes I + I \otimes L_1\f$ and \f$L_1\f$ is the
\f$N \times N\f$ tridiagonal matrix \f$\mathrm{tridiag}(-1,\,2,\,-1)\f$.

---

## Eigenvalue decomposition {#sec_poisson_eigen}

\f$L_1\f$ is symmetric and its eigenvectors are the sine modes

\f[
  F_{jk} = \sin\!\left(\frac{jk\pi}{N+1}\right), \quad j,k = 1,\ldots,N,
\f]

so \f$L_1 = F\,D\,F^T\f$ with diagonal eigenvalues

\f[
  D_k = 2\left(1 - \cos\frac{k\pi}{N+1}\right), \quad k = 1,\ldots,N.
\f]

Because \f$L_2 = L_1 \otimes I + I \otimes L_1\f$, its eigenvalues are
\f$D_j + D_k\f$ for all pairs \f$(j,k)\f$.  Applying the 2-D DST to both sides of
the discrete system decouples it into \f$N^2\f$ independent scalar equations:

\f[
  (D_j + D_k)\,\hat{u}_{jk} = h^2\,\hat{f}_{jk}.
\f]

---

## Algorithm (FD solver) {#sec_poisson_fd}

1. Compute \f$\hat{F} = \mathrm{DST2D}(h^2 f)\f$.
2. Divide pointwise: \f$\hat{U}_{jk} = \hat{F}_{jk} / (D_j + D_k)\f$.
3. Recover \f$u = \mathrm{IDST2D}(\hat{U})\f$.

The inverse DST-I satisfies \f$\mathrm{IDST} = \tfrac{2}{N+1}\,\mathrm{DST}\f$, so
step 3 is another forward DST followed by scaling by \f$\bigl(\tfrac{2}{N+1}\bigr)^2\f$.

**Error**: \f$O(h^2)\f$, from the FD truncation in the eigenvalue
\f$D_k = 2(1-\cos(k\pi/(N+1)))\f$.  The ratio \f$\|u_h - u\|_\infty / \|u_{h/2} - u\|_\infty \to 4\f$
as \f$h \to 0\f$.

---

## Algorithm (spectral solver) {#sec_poisson_spectral}

Replace the FD eigenvalue \f$D_k\f$ with the exact eigenvalue \f$(k\pi)^2\f$ of the
continuous operator \f$-d^2/dx^2\f$.

**Normalization.** The unnormalised DST satisfies \f$F \cdot F = \tfrac{N+1}{2}\,I\f$,
so the continuous spectral coefficient of \f$f\f$ is

\f[
  \tilde{f}_{jk} = \frac{4}{(N+1)^2}\,\hat{F}_{j-1,\,k-1}.
\f]

Dividing by \f$(j^2+k^2)\pi^2\f$ and expanding the reconstruction sum shows that
\f$u = \mathrm{DST2D}(\hat{U})\f$ without an extra \f$\bigl(\tfrac{2}{N+1}\bigr)^2\f$
factor, where

\f[
  \hat{U}_{jk} = \frac{4}{(N+1)^2\,\pi^2\,(j^2+k^2)}\,\hat{F}_{jk}.
\f]

**Algorithm:**

1. Compute \f$\hat{F} = \mathrm{DST2D}(f)\f$ (no \f$h^2\f$ prefactor).
2. Multiply: \f$\hat{U}_{jk} = \hat{F}_{jk} \cdot \dfrac{4}{(N+1)^2\,\pi^2\,(j^2+k^2)}\f$.
3. Recover \f$u = \mathrm{DST2D}(\hat{U})\f$.

**Error**: zero up to floating-point rounding for any \f$f\f$ expressible as a finite
sum of DST eigenfunctions at the grid resolution.  For smooth \f$f\f$ with many active
modes, convergence is exponential in \f$N\f$.

---

## DST-I via complex FFT {#sec_poisson_dst}

The unnormalised Discrete Sine Transform of type I of an \f$N\f$-point real vector
\f$x\f$ is

\f[
  X[k] = \sum_{j=1}^{N} x[j]\,\sin\!\left(\frac{jk\pi}{N+1}\right), \quad k = 1,\ldots,N.
\f]

It is computed via a complex FFT on the odd-extended sequence of length \f$M = 2(N+1)\f$:

\f[
  y = \bigl[0,\; x[1],\; \ldots,\; x[N],\; 0,\; -x[N],\; \ldots,\; -x[1]\bigr].
\f]

The FFT of \f$y\f$ satisfies \f$Y[k] = -2i\,X[k]\f$, so

\f[
  X[k] = -\frac{1}{2}\,\mathrm{Im}\bigl(Y[k]\bigr), \quad k = 1,\ldots,N.
\f]

**Size constraint**: \f$M = 2(N+1)\f$ must be a power of two for the radix-2 backend,
which requires \f$N = 2^p - 1\f$ (e.g. 7, 15, 31, 63, 127, 255).

The 2-D DST is \f$F^T \cdot A \cdot F\f$, applied by sweeping DST-I over columns
then rows.  The total cost is \f$O(N^2 \log N)\f$.

---

## API reference {#sec_poisson_api}

```cpp
namespace num::pde {

/// Solve -Delta u = f on (0,1)^2 via DST with FD eigenvalues.  Error O(h^2).
/// N must satisfy N = 2^p - 1.
[[nodiscard]] Matrix poisson2d_fd(const Matrix& f, int N);

/// Solve -Delta u = f on (0,1)^2 via DST with exact eigenvalues.
/// Machine-precision error for f in the DST eigenbasis.
/// N must satisfy N = 2^p - 1.
[[nodiscard]] Matrix poisson2d(const Matrix& f, int N);

} // namespace num::pde
```

Both functions throw `std::invalid_argument` if \f$N \leq 0\f$ or
\f$N \neq 2^p - 1\f$.

The `f` matrix uses row-major indexing: `f(i, j)` is the RHS value at
\f$(x,y) = ((i+1)h,\,(j+1)h)\f$.

---

## Worked example {#sec_poisson_example}

Test problem with two active modes:

\f[
  f(x,y) = 2\pi^2\sin(\pi x)\sin(\pi y) + 5\pi^2\sin(2\pi x)\sin(\pi y),
\f]
\f[
  u(x,y) = \sin(\pi x)\sin(\pi y) + \sin(2\pi x)\sin(\pi y).
\f]

```cpp
#include "numerics.hpp"
#include <cmath>

using namespace num;

int main() {
    const int N = 31;               // 2^5 - 1
    const double h = 1.0 / (N + 1);
    const double pi = M_PI;

    Matrix f(N, N), u_exact(N, N);
    for (int i = 0; i < N; ++i) {
        const double x = (i + 1) * h;
        for (int j = 0; j < N; ++j) {
            const double y = (j + 1) * h;
            f(i, j) = (2*pi*pi*std::sin(pi*x) + 5*pi*pi*std::sin(2*pi*x)) * std::sin(pi*y);
            u_exact(i, j) = (std::sin(pi*x) + std::sin(2*pi*x)) * std::sin(pi*y);
        }
    }

    Matrix u_fd   = pde::poisson2d_fd(f, N);   // O(h^2) error
    Matrix u_spec = pde::poisson2d(f, N);       // machine precision
}
```

Convergence table (max-norm error, \f$N = 2^p - 1\f$):

| N   | h        | FD error   | ratio | Spectral error |
|-----|----------|------------|-------|----------------|
| 3   | 0.250000 | 2.303e-01  | —     | 3.9e-16        |
| 7   | 0.125000 | 5.392e-02  | 4.27  | 3.9e-16        |
| 15  | 0.062500 | 1.327e-02  | 4.06  | 5.6e-16        |
| 31  | 0.031250 | 3.304e-03  | 4.02  | 8.9e-16        |
| 63  | 0.015625 | 8.285e-04  | 3.99  | 1.3e-15        |
| 127 | 0.007812 | 2.071e-04  | 4.00  | 1.1e-15        |
| 255 | 0.003906 | 5.176e-05  | 4.00  | 1.1e-15        |

The FD ratio converges to 4, confirming \f$O(h^2)\f$.  The spectral error stays at
the level of floating-point rounding for all \f$N\f$, because this \f$f\f$ has only
two active DST modes: \f$(j,k) = (1,1)\f$ and \f$(2,1)\f$.
