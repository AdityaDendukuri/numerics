# Root Finding {#page_roots}

Bracketing and derivative-based solvers for scalar nonlinear equations \f$f(x) = 0\f$.

---

## 1. Root Result (num::RootResult)

```cpp
struct RootResult {
    double root{0.0};     // Computed root x*
    double residual{0.0}; // Residual |f(x*)|
    idx iterations{0};    // Iteration count performed
    bool converged{false};// True if |f(x*)| <= tol or bracket size <= tol
};
```

---

## 2. Solvers

### Bisection Method (num::bisection)
Linear convergence on a bracketing interval \f$[a, b]\f$ with \f$f(a) \cdot f(b) < 0\f$.

```cpp
RootResult bisection(ScalarFn f, double a, double b, double tol = 1e-8, idx max_iter = 100);
```

```cpp
auto f = [](double x) { return std::cos(x) - x; };
num::RootResult r = num::bisection(f, 0.0, 1.0, 1e-12);
```

### Brent's Method (num::brent)
Brent's method (bisection, secant, and inverse quadratic interpolation).

```cpp
RootResult brent(ScalarFn f, double a, double b, double tol = 1e-8, idx max_iter = 100);
```

```cpp
num::RootResult r = num::brent(f, 0.0, 1.0, 1e-12);
```

### Secant Method (num::secant)
Derivative-free quasi-Newton iteration from two initial guesses.

```cpp
RootResult secant(ScalarFn f, double x0, double x1, double tol = 1e-8, idx max_iter = 100);
```

```cpp
num::RootResult r = num::secant(f, 0.0, 1.0, 1e-12);
```

### Newton-Raphson (num::newton)
Newton–Raphson iteration with analytical derivative \f$f'(x)\f$.

```cpp
RootResult newton(ScalarFn f, ScalarFn df, double x0, double tol = 1e-8, idx max_iter = 100);
```

```cpp
auto f  = [](double x) { return x * x - 2.0; };
auto df = [](double x) { return 2.0 * x; };
num::RootResult r = num::newton(f, df, 1.0, 1e-12);
```

---

## Complete Example

@example 08_root_finding_and_quadrature.cpp

