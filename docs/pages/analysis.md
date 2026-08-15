# Analysis Examples {#page_analysis}

Analysis routines provide scalar quadrature and root finding. They are intended
for small numerical subproblems inside larger solvers and experiments.

## Composite Quadrature

```cpp
#include <numerics.hpp>

auto f = [](double x) {
    return std::exp(-x * x);
};

double q1 = num::trapz(f, 0.0, 1.0, 4096);
double q2 = num::simpson(f, 0.0, 1.0, 4096);
double q3 = num::romberg(f, 0.0, 1.0, 1e-12);
```

`Backend::omp` parallelizes the panel sums in `trapz` and `simpson`.

```cpp
double q = num::simpson(f, 0.0, 1.0, 1 << 20, num::Backend::omp);
```

## Gauss-Legendre Rule

```cpp
auto p = [](double x) {
    return x * x * x * x;
};

double exact_for_degree_4 = num::gauss_legendre(p, -1.0, 1.0, 3);
```

With `p` points, the rule is exact for polynomials through degree \(2p-1\).

## Bracketed Roots

```cpp
auto g = [](double x) {
    return std::cos(x) - x;
};

num::RootResult r = num::brent(g, 0.0, 1.0, 1e-12);
```

Use Brent's method when a sign-changing bracket is available.

## Newton Iteration

```cpp
auto f0 = [](double x) { return x * x - 2.0; };
auto df = [](double x) { return 2.0 * x; };

num::RootResult r = num::newton(f0, df, 1.0, 1e-12);
```

Newton iteration is appropriate when the derivative is available and the
starting value is in the local basin of attraction.
