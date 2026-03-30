# Root Finding {#page_roots_notes}

Given a continuous function \f$f : \mathbb{R} \to \mathbb{R}\f$, find \f$x^*\f$ such that \f$f(x^*) = 0\f$.

**Implementation**: `include/analysis/roots.hpp`, `src/analysis/roots.cpp`

---

## Bisection

The Intermediate Value Theorem guarantees that if \f$f(a) \cdot f(b) < 0\f$, there exists a root in \f$[a, b]\f$.
Bisection exploits this by repeatedly halving the interval.

**Algorithm**:
```
given [a, b] with f(a)*f(b) < 0
repeat:
    mid = (a + b) / 2
    if f(mid) ~= 0 or (b - a)/2 < tol: return mid
    if f(a)*f(mid) < 0: b = mid
    else:                a = mid
```

**Convergence**: Linear. After \f$n\f$ iterations, the error is at most \f$(b - a) / 2^n\f$.

\f[|x_n - x^*| \leq \frac{b - a}{2^n}\f]

To achieve tolerance \f$\varepsilon\f$: \f$n \geq \log_2\!\left(\frac{b-a}{\varepsilon}\right)\f$ iterations.

**Properties**: Guaranteed to converge. No derivative required. Slow.

---

## Newton-Raphson

Expand \f$f\f$ around the current estimate \f$x_k\f$ via Taylor:

\f[f(x^*) = f(x_k) + f'(x_k)(x^* - x_k) + O((x^* - x_k)^2) = 0\f]

Dropping higher-order terms and solving for \f$x^*\f$:

\f[x_{k+1} = x_k - \frac{f(x_k)}{f'(x_k)}\f]

**Convergence**: Quadratic near the root. If \f$e_k = x_k - x^*\f$:

\f[|e_{k+1}| \approx \frac{|f''(x^*)|}{2|f'(x^*)|} |e_k|^2\f]

In practice, the number of correct decimal digits roughly doubles each iteration.

**Properties**: Fast near the root. Requires \f$f'\f$. Can diverge if initial guess is poor or \f$f'(x_k) \approx 0\f$.

---

## Secant Method

Newton without the derivative. Approximate \f$f'(x_k)\f$ by a finite difference using the two most recent iterates:

\f[x_{k+1} = x_k - f(x_k) \cdot \frac{x_k - x_{k-1}}{f(x_k) - f(x_{k-1})}\f]

**Convergence**: Superlinear with order \f$\phi \approx 1.618\f$ (golden ratio):

\f[|e_{k+1}| \approx C |e_k|^\phi\f]

Slower than Newton but does not require a derivative.

**Properties**: Needs two initial points. Can stagnate if \f$f(x_k) \approx f(x_{k-1})\f$.

---

## Brent's Method

Combines bisection (guaranteed), secant (fast), and inverse quadratic interpolation (faster) in a single algorithm. At each step, it attempts the faster interpolation methods; if the proposed step is outside the current bracket or too large, it falls back to bisection.

**Inverse quadratic interpolation**: Given three points \f$(a, f(a)), (b, f(b)), (c, f(c))\f$, fit a quadratic through the three points in the \f$x\f$-direction (treating \f$x\f$ as a function of \f$f\f$) and evaluate at \f$f = 0\f$.

**Invariant**: \f$[b, c]\f$ always brackets the root. \f$b\f$ is always the current best estimate.

**Convergence**: Superlinear in practice, with guaranteed worst-case linear convergence (from bisection fallback).

**Properties**: This is the recommended method when a bracket is available. No derivative required. Used in most production software (Brent 1973, also the basis of `scipy.optimize.brentq`).

---

## Comparison

| Method | Convergence | Needs bracket | Needs \f$f'\f$ | Recommended when |
|---|---|---|---|---|
| Bisection | Linear | Yes | No | Robustness required, slow OK |
| Newton | Quadratic | No | Yes | Good initial guess, \f$f'\f$ available |
| Secant | Superlinear | No | No | \f$f'\f$ unavailable, two initial points known |
| Brent | Superlinear (guaranteed linear) | Yes | No | General purpose |

---

## Parallel Structure

Scalar root finding is inherently sequential -- each iterate depends on the previous one.

**Where parallelism appears**:

- **Multiple independent roots**: if \f$f\f$ has \f$k\f$ roots on \f$[a, b]\f$, each can be bracketed and solved in parallel.
- **Multivariate Newton** (not yet implemented): each Newton step requires solving \f$J \Delta x = -f(x)\f$, where \f$J\f$ is the Jacobian. Computing \f$J\f$ (finite differences) is embarrassingly parallel across columns.
- **Parameter sweeps**: solving \f$f(x; \lambda) = 0\f$ for many values of parameter \f$\lambda\f$ is trivially parallel.

---

## Exercises

1. Bisection requires \f$n = \lceil \log_2((b-a)/\varepsilon) \rceil\f$ iterations for tolerance \f$\varepsilon\f$. How many iterations are needed to find \f$\sqrt{2}\f$ to 15 significant figures starting from \f$[1, 2]\f$?

2. Show that Newton's method applied to \f$f(x) = x^2 - c\f$ gives the Babylonian square root algorithm \f$x_{k+1} = \frac{1}{2}(x_k + c/x_k)\f$.

3. Newton's method can cycle on \f$f(x) = x^3 - 2x + 2\f$ starting from \f$x_0 = 0\f$. Show why, and suggest a fix.

4. (Parallel) You need to find roots of \f$f(x; \lambda) = \lambda x^3 - x - 1 = 0\f$ for 1024 values of \f$\lambda \in [0.5, 2.0]\f$. Describe an OpenMP strategy. What is the expected parallel speedup?
