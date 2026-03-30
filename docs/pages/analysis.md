# Numerical Analysis {#page_analysis}

---

## Numerical Integration (Quadrature) {#sec_quadrature}

Quadrature methods approximate \f$I = \int_a^b f(x)\,dx\f$ when an antiderivative is
unavailable or expensive. The choice of method depends on the smoothness of \f$f\f$, the
required accuracy, and whether evaluations are costly.

### Newton-Cotes Rules

Newton-Cotes methods approximate \f$f\f$ by a polynomial interpolant on equally-spaced
nodes and integrate the polynomial exactly.

#### Trapezoidal Rule

Partition \f$[a,b]\f$ into \f$n\f$ panels of width \f$h = (b-a)/n\f$ and approximate
\f$f\f$ by a piecewise linear interpolant:

\f[
  T_n = h\left[\frac{f(a)+f(b)}{2} + \sum_{i=1}^{n-1} f(a+ih)\right].
\f]

**Error bound** (\f$O(h^2)\f$ globally):

\f[
  |I - T_n| \leq \frac{(b-a)^3}{12\,n^2}\max_{x\in[a,b]}|f''(x)|.
\f]

Doubling \f$n\f$ reduces the error by a factor of 4.

#### Simpson's 1/3 Rule

Approximate \f$f\f$ by a piecewise quadratic on pairs of panels (\f$n\f$ must be even):

\f[
  S_n = \frac{h}{3}\left[f(a)+f(b)
        + 4\sum_{\text{odd }i}f(a+ih)
        + 2\sum_{\text{even }i}f(a+ih)\right].
\f]

**Error bound** (\f$O(h^4)\f$ globally):

\f[
  |I - S_n| \leq \frac{(b-a)^5}{180\,n^4}\max_{x\in[a,b]}|f^{(4)}(x)|.
\f]

Simpson's rule is exact for polynomials of degree \f$\leq 3\f$ (one order higher than
expected from a degree-2 interpolant, due to the symmetric placement of nodes).

### Gauss-Legendre Quadrature

Gauss-Legendre optimizes **both** nodes and weights to maximize the polynomial degree
integrated exactly. With \f$p\f$ points it is exact for polynomials up to degree
\f$2p - 1\f$:

\f[
  \int_{-1}^{1} f(x)\,dx \approx \sum_{i=1}^{p} w_i f(x_i),
\f]

where \f$x_i\f$ are the roots of the \f$p\f$-th Legendre polynomial \f$P_p(x)\f$ and the
weights \f$w_i\f$ are the corresponding Christoffel numbers.

For a general interval \f$[a,b]\f$, apply the change of variables
\f$x = \tfrac{a+b}{2} + \tfrac{b-a}{2}\,t\f$:

\f[
  \int_a^b f(x)\,dx
  = \frac{b-a}{2}\sum_{i=1}^{p} w_i\, f\!\left(\frac{a+b}{2}+\frac{b-a}{2}\,x_i\right).
\f]

A \f$p\f$-point Gauss-Legendre rule achieves the accuracy of a degree-\f$(2p-1)\f$ fit,
whereas a \f$p\f$-point trapezoidal rule achieves only \f$O(h^2) = O(1/p^2)\f$ accuracy.

| \f$p\f$ | Polynomial degree exact | Nodes |
|---------|------------------------|-------|
| 1 | 1 | 0 |
| 2 | 3 | \f$\pm 1/\sqrt{3}\f$ |
| 3 | 5 | 0, \f$\pm\sqrt{3/5}\f$ |
| 4 | 7 | 4 non-trivial nodes |
| 5 | 9 | 5 non-trivial nodes |

### Adaptive Simpson

Fixed-panel methods concentrate evaluations uniformly regardless of local behavior.
Adaptive methods refine only where the error estimate is large.

Given an interval \f$[a,b]\f$ with midpoint \f$m = (a+b)/2\f$, the error estimator is

\f[
  \text{error} \approx \frac{S(a,b) - S(a,m) - S(m,b)}{15},
\f]

where \f$S(\cdot,\cdot)\f$ denotes the single-panel Simpson estimate. The factor \f$1/15\f$
comes from Richardson extrapolation applied to the \f$O(h^4)\f$ error of Simpson's rule:
combining the coarse and fine estimates eliminates the leading error term and the
remainder is \f$1/15\f$ of the discrepancy.

If \f$|\delta| > 15\varepsilon\f$ the interval is bisected and the process recurses.
Evaluations automatically concentrate near singularities and regions of rapid variation.

### Romberg Integration

Romberg applies Richardson extrapolation to the trapezoidal rule. The Euler-Maclaurin
expansion shows that the trapezoidal error has an asymptotic expansion in even powers of
\f$h\f$:

\f[
  T(h) = I + c_2 h^2 + c_4 h^4 + c_6 h^6 + \cdots
\f]

Combining estimates at \f$h\f$ and \f$h/2\f$ eliminates the \f$c_2 h^2\f$ term. The
general Richardson recurrence builds a triangular table:

\f[
  R_{i,j} = \frac{4^j R_{i,j-1} - R_{i-1,j-1}}{4^j - 1}.
\f]

The diagonal entry \f$R_{k,k}\f$ is exact for polynomials of degree up to \f$2k-1\f$ and
converges faster than any fixed power of \f$h\f$ for smooth \f$f\f$ (super-algebraic
convergence). For analytic \f$f\f$, convergence is exponential.

### Method Comparison

| Method | Error order | Evaluations | Best for |
|--------|------------|-------------|----------|
| Trapezoidal | \f$O(h^2)\f$ | \f$n+1\f$ | Baseline, periodic \f$f\f$ |
| Simpson | \f$O(h^4)\f$ | \f$n+1\f$ | Smooth \f$f\f$, low cost |
| Gauss-Legendre (\f$p\f$ pts) | Exact to degree \f$2p-1\f$ | \f$p\f$ | Smooth \f$f\f$, few evaluations |
| Adaptive Simpson | Error-controlled | Automatic | Irregular or oscillatory \f$f\f$ |
| Romberg | Super-algebraic | \f$O(2^k)\f$ | Very smooth \f$f\f$, high accuracy |

**API**: @ref num::trapz, @ref num::simpson, @ref num::gauss_legendre,
@ref num::adaptive_simpson, @ref num::romberg

---

## Root Finding {#sec_roots}

Given a continuous function \f$f : \mathbb{R} \to \mathbb{R}\f$, find \f$x^*\f$ such that
\f$f(x^*) = 0\f$. Methods differ in convergence speed, the information required (bracket,
derivative), and robustness guarantees.

### Bisection

The Intermediate Value Theorem guarantees a root in \f$[a,b]\f$ when \f$f(a)f(b) < 0\f$.
Bisection exploits this by repeatedly halving the bracket.

**Error bound**: after \f$n\f$ iterations,

\f[
  |x_n - x^*| \leq \frac{b-a}{2^n}.
\f]

To reach tolerance \f$\varepsilon\f$: \f$n \geq \log_2\!\left(\tfrac{b-a}{\varepsilon}\right)\f$
iterations are required. Convergence is guaranteed and derivative-free, but the linear
rate makes bisection slow for tight tolerances.

**API**: @ref num::bisection

### Newton-Raphson

A first-order Taylor expansion of \f$f\f$ around \f$x_k\f$ and setting the linearization
to zero gives

\f[
  x_{k+1} = x_k - \frac{f(x_k)}{f'(x_k)}.
\f]

**Quadratic convergence** near the root: if \f$e_k = x_k - x^*\f$,

\f[
  |e_{k+1}| \approx \frac{|f''(x^*)|}{2|f'(x^*)|}\,|e_k|^2.
\f]

The number of correct decimal digits roughly doubles each iteration. The method can
diverge from a poor starting guess or when \f$f'(x_k) \approx 0\f$.

**API**: @ref num::newton

### Secant Method

The secant method approximates \f$f'\f$ by a finite difference of the two most recent
iterates:

\f[
  x_{k+1} = x_k - f(x_k)\cdot\frac{x_k - x_{k-1}}{f(x_k) - f(x_{k-1})}.
\f]

**Superlinear convergence** with order \f$\phi = (1+\sqrt{5})/2 \approx 1.618\f$ (the
golden ratio):

\f[
  |e_{k+1}| \approx C\,|e_k|^\phi.
\f]

The method requires no derivative but needs two initial points and can stagnate if
\f$f(x_k) \approx f(x_{k-1})\f$.

**API**: @ref num::secant

### Brent's Method

Brent's method combines three strategies in a single algorithm:

- **Bisection** -- always guaranteed to contract the bracket.
- **Secant step** -- superlinear acceleration.
- **Inverse quadratic interpolation** -- fit a quadratic through three points in the
  \f$x\f$-direction (treating \f$x\f$ as a function of \f$f\f$) and evaluate at \f$f = 0\f$.

At each step Brent attempts the faster interpolation method; if the proposed step falls
outside the current bracket or is too large, it falls back to bisection. The bracket
\f$[b,c]\f$ always contains the root and \f$b\f$ is the current best estimate.

**Convergence**: superlinear in practice with a guaranteed worst-case linear rate from the
bisection fallback. No derivative is required. This is the recommended general-purpose
method when a bracket is available, and is the basis of `scipy.optimize.brentq`.

**API**: @ref num::brent

### Method Comparison

| Method | Convergence | Needs bracket | Needs \f$f'\f$ | Use when |
|--------|------------|---------------|----------------|----------|
| Bisection | Linear | Yes | No | Robustness required, speed not critical |
| Newton | Quadratic | No | Yes | Good initial guess, \f$f'\f$ available |
| Secant | Superlinear (\f$\phi \approx 1.618\f$) | No | No | \f$f'\f$ unavailable, two initial points known |
| Brent | Superlinear (guaranteed linear) | Yes | No | General purpose |
