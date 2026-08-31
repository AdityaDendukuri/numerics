# Quadrature {#page_quadrature}

Composite, Gaussian, adaptive, Romberg, and Talbot contour quadrature.

---

## 1. Integrand Signature

Callables conforming to `num::ScalarFn` (e.g. `double(double)` lambdas):

```cpp
#include <numerics.hpp>

auto gaussian = [](double x) { return std::exp(-x * x); };
```

---

## 2. Composite Rules

### num::trapz
Composite trapezoidal rule: \f$\int_a^b f(x)\,dx \approx \frac{h}{2} [f(a) + 2\sum_{i=1}^{n-1} f(x_i) + f(b)] + \mathcal{O}(h^2)\f$.

```cpp
double trapz(num::ScalarFn f, double a, double b, num::idx n,
             auto backend = num::backend::dflt);
```

```cpp
double q = num::trapz(gaussian, 0.0, 1.0, 4096);
```

### num::simpson
Composite Simpson's 1/3 rule (\f$\mathcal{O}(h^4)\f$). Requires even panel count \f$n\f$.

```cpp
double simpson(num::ScalarFn f, double a, double b, num::idx n,
               auto backend = num::backend::dflt);
```

```cpp
double q = num::simpson(gaussian, 0.0, 1.0, 4096);
double q_omp = num::simpson(gaussian, 0.0, 1.0, 1 << 20, num::backend::omp); // OpenMP
```

---

## 3. Gaussian Quadrature (num::gauss_legendre)

Exact for polynomials of degree up to \f$2n - 1\f$. Supports \f$n \in [1, 5]\f$ nodes.

```cpp
double gauss_legendre(num::ScalarFn f, double a, double b, int n = 5);
```

```cpp
auto quartic = [](double x) { return x * x * x * x; };
double q = num::gauss_legendre(quartic, -1.0, 1.0, 3); // Exact for polynomials up to degree 5
```

---

## 4. Adaptive and Extrapolated Rules

### num::adaptive_simpson
Recursively subdivides panels where estimated local error exceeds \f$\varepsilon\f$.

```cpp
double adaptive_simpson(num::ScalarFn f, double a, double b, double tol = 1e-10);
```

```cpp
double q = num::adaptive_simpson(gaussian, 0.0, 1.0, 1e-10);
```

### num::romberg
Richardson extrapolation over composite trapezoidal estimates.

```cpp
double romberg(num::ScalarFn f, double a, double b, double tol = 1e-12, int max_depth = 12);
```

```cpp
double q = num::romberg(gaussian, 0.0, 1.0, 1e-12, 12);
```

---

## 5. Inverse Laplace Contour Integration (num::TalbotQuadrature)

Evaluates the Bromwich contour integral \f$f(t) = \mathcal{L}^{-1}[F](t) \approx \sum_{k=1}^M w_k F(z_k)\f$ along the Weideman–Talbot hyperbolic contour:

```cpp
num::TalbotQuadrature contour{/*n_nodes=*/16};
auto nodes = contour.nodes(/*t=*/1.0); // Precomputed complex nodes and weights
```

---

## Complete Examples

- @example 08_root_finding_and_quadrature.cpp
- @example 13_talbot_spectral_validation.cpp

