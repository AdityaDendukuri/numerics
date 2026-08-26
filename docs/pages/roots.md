# Root Finding {#page_roots}

The `roots` module provides bracketing and derivative-based methods for scalar equations \f$f(x) = 0\f$.

---

## 1. Root Finding

Solves the nonlinear scalar equation \f$f(x^*) = 0\f$.

### Bisection Method

Guaranteed linear convergence for continuous functions on a bracketing interval \f$[a, b]\f$ where \f$f(a) \cdot f(b) < 0\f$:

\f[
x_{k+1} = \frac{a_k + b_k}{2}, \qquad |x_{k+1} - x^*| \le \frac{b - a}{2^{k+1}}
\f]

```cpp
auto f = [](double x) { return std::cos(x) - x; };
num::RootResult root = num::bisection(f, 0.0, 1.0, 1e-12); // Bracket must change sign.
```

### Brent's Method

Robust hybrid algorithm combining root bisection, linear secant interpolation, and inverse quadratic interpolation:

```cpp
num::RootResult root = num::brent(f, 0.0, 1.0, 1e-12); // Fast superlinear convergence with bracket safety.
```

### Secant Method

Derivative-free quasi-Newton iteration requiring two initial guesses:

\f[
x_{k+1} = x_k - f(x_k)\,\frac{x_k - x_{k-1}}{f(x_k) - f(x_{k-1})}
\f]

```cpp
num::RootResult root = num::secant(f, 0.0, 1.0, 1e-12);
```

### Newton–Raphson Method

Quadratic convergence using explicit analytical derivatives \f$f'(x)\f$:

\f[
x_{k+1} = x_k - \frac{f(x_k)}{f'(x_k)}
\f]

```cpp
auto f = [](double x) { return x * x - 2.0; };
auto df = [](double x) { return 2.0 * x; };

num::RootResult root = num::newton(f, df, 1.0, 1e-12); // Explicit derivative supplied.
```

### Root Metadata & Convergence Diagnostics

```cpp
if (!root.converged) {
    throw std::runtime_error("Root solver failed to converge within maximum iterations");
}

double x = root.root;            // Computed root x*
double error = root.residual;    // Absolute residual |f(x*)|
num::idx work = root.iterations; // Iteration count performed
```

---

## Complete Example

@example 08_root_finding_and_quadrature.cpp
