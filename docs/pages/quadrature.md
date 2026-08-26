# Quadrature {#page_quadrature}

The `quadrature` module provides composite and Gaussian rules, adaptive refinement, Richardson extrapolation, and contour quadrature for inverse Laplace transforms.

---

## 1. Integrands & Callable Signatures

Scalar integrands are passed as callables conforming to `num::ScalarFn` (e.g. `std::function<double(double)>` or lambdas):

```cpp
#include <numerics.hpp>
#include <cmath>

num::ScalarFn gaussian = [](double x) {
    return std::exp(-x * x); // f(x) = exp(-x^2)
};
```

---

## 2. Composite Quadrature

### Trapezoidal Rule

Approximates the definite integral over \f$[a, b]\f$ divided into \f$n\f$ uniform panels with step size \f$h = \frac{b - a}{n}\f$:

\f[
\int_a^b f(x)\,dx \approx \frac{h}{2} \left[ f(a) + 2\sum_{i=1}^{n-1} f(x_i) + f(b) \right] + \mathcal{O}(h^2)
\f]

```cpp
double q = num::trapz(gaussian, 0.0, 1.0, 4096); // 4096 trapezoidal panels.
```

### Simpson's Rule

Fits parabolic arcs across pairs of panels (requires an even panel count \f$n\f$):

\f[
\int_a^b f(x)\,dx \approx \frac{h}{3} \left[ f(a) + 4\sum_{i=1,3,\dots}^{n-1} f(x_i) + 2\sum_{i=2,4,\dots}^{n-2} f(x_i) + f(b) \right] + \mathcal{O}(h^4)
\f]

```cpp
double q = num::simpson(gaussian, 0.0, 1.0, 4096); // Panel count must be even.
```

```cpp
// Parallel summation using OpenMP backend across large panel grids
double q = num::simpson(gaussian, 0.0, 1.0, 1 << 20, num::backend::omp);
```

---

## 3. Gaussian Quadrature

Gauss–Legendre quadrature evaluates integrals with optimal algebraic degree of exactness, integrating polynomials of degree up to \f$2n - 1\f$ exactly:

\f[
\int_{-1}^1 f(x)\,dx \approx \sum_{i=1}^n w_i f(x_i)
\f]

```cpp
auto quartic = [](double x) { return x * x * x * x; };
double q = num::gauss_legendre(quartic, -1.0, 1.0, 3); // Exact for polynomials through degree 5.
```

`num::gauss_legendre` supports \f$n \in [1, 5]\f$ quadrature nodes with precomputed roots and weights.

---

## 4. Adaptive Quadrature & Richardson Extrapolation

### Adaptive Simpson's Method

Recursively refines panel subdivisions where the local error estimate exceeds tolerance \f$\varepsilon\f$:

```cpp
double q = num::adaptive_simpson(gaussian, 0.0, 1.0, 1e-10); // Automatically refines steep gradients.
```

### Romberg Integration

Combines composite trapezoidal evaluations with Richardson extrapolation:

\f[
R(k, m) = \frac{4^m R(k, m-1) - R(k-1, m-1)}{4^m - 1}
\f]

```cpp
double q = num::romberg(gaussian, 0.0, 1.0, 1e-12, 12); // At most 12 extrapolation levels.
```

---

## 5. Inverse Laplace Contour Integration (Talbot Quadrature)

Evaluates the Bromwich inverse Laplace integral for \f$t > 0\f$:

\f[
f(t) = \mathcal{L}^{-1}[F](t) = \frac{1}{2\pi i} \oint_\Gamma e^{zt} F(z)\,dz \approx \sum_{k=1}^M w_k F(z_k)
\f]

where \f$\Gamma\f$ is deformed along the Weideman–Talbot hyperbolic contour:

```cpp
num::TalbotQuadrature contour{16}; // 16 complex contour nodes
auto nodes = contour.nodes(1.0);   // Scaled for target time t = 1.0
```

---
