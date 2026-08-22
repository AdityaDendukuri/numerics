# Analysis Examples {#page_analysis}

## Integrands

```cpp
#include <numerics.hpp>
#include <cmath>

num::ScalarFn gaussian = [](double x) {
    return std::exp(-x * x); // Callable stored with a uniform scalar signature.
};
```

## Composite Quadrature

```cpp
double q = num::trapz(gaussian, 0.0, 1.0, 4096); // 4096 trapezoidal panels.
```

```cpp
double q = num::simpson(gaussian, 0.0, 1.0, 4096); // Panel count must be even.
```

```cpp
double q = num::simpson(
    gaussian, 0.0, 1.0, 1 << 20, num::Backend::omp); // Parallel panel sum.
```

## Gaussian Quadrature

```cpp
auto quartic = [](double x) { return x * x * x * x; };
double q = num::gauss_legendre(quartic, -1.0, 1.0, 3); // Exact through degree five.
```

`gauss_legendre` accepts between one and five quadrature points.

## Adaptive Quadrature

```cpp
double q = num::adaptive_simpson(gaussian, 0.0, 1.0, 1e-10); // Refine where needed.
```

```cpp
double q = num::romberg(gaussian, 0.0, 1.0, 1e-12, 12); // At most 12 refinement levels.
```

## Bisection

```cpp
auto f = [](double x) { return std::cos(x) - x; };
num::RootResult root = num::bisection(f, 0.0, 1.0, 1e-12); // Bracket must change sign.
```

## Brent's Method

```cpp
num::RootResult root = num::brent(f, 0.0, 1.0, 1e-12); // Bracketed superlinear solve.
```

## Secant Method

```cpp
num::RootResult root = num::secant(f, 0.0, 1.0, 1e-12); // Uses two initial values.
```

## Newton's Method

```cpp
auto f = [](double x) { return x * x - 2.0; };
auto df = [](double x) { return 2.0 * x; };

num::RootResult root = num::newton(f, df, 1.0, 1e-12); // Derivative supplied explicitly.
```

## Root Metadata

```cpp
if (!root.converged) {
    throw std::runtime_error("root solve failed");
}

double x = root.root;            // Final root estimate.
double error = root.residual;    // Absolute function residual.
num::idx work = root.iterations; // Iterations performed.
```

## Complete Program

@example 08_root_finding_and_quadrature.cpp
