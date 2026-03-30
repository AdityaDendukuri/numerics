# Numerical Quadrature {#page_quadrature_notes}

Compute \f$I = \int_a^b f(x)\, dx\f$ when an antiderivative is unavailable or expensive.

**Implementation**: `include/analysis/quadrature.hpp`, `src/analysis/quadrature.cpp`

---

## Newton-Cotes Rules

All Newton-Cotes methods approximate \f$f\f$ by a polynomial interpolant on equally-spaced points and integrate the polynomial exactly.

### Trapezoidal Rule

Approximate \f$f\f$ by a piecewise linear interpolant on \f$n\f$ equally-spaced panels of width \f$h = (b-a)/n\f$:

\f[\int_a^b f(x)\, dx \approx h \left[ \frac{f(a) + f(b)}{2} + \sum_{i=1}^{n-1} f(a + ih) \right]\f]

**Error**: \f$O(h^2)\f$ per panel, \f$O(h^2)\f$ globally. Specifically:

\f[\left| I - T_n \right| \leq \frac{(b-a)^3}{12 n^2} \max_{x \in [a,b]} |f''(x)|\f]

Doubling \f$n\f$ reduces error by a factor of 4.

### Simpson's Rule

Approximate \f$f\f$ by a piecewise quadratic on pairs of panels (\f$n\f$ must be even):

\f[\int_a^b f(x)\, dx \approx \frac{h}{3} \left[ f(a) + f(b) + 4\sum_{\text{odd } i} f(a+ih) + 2\sum_{\text{even } i} f(a+ih) \right]\f]

**Error**: \f$O(h^4)\f$ globally.

\f[\left| I - S_n \right| \leq \frac{(b-a)^5}{180 n^4} \max_{x \in [a,b]} |f^{(4)}(x)|\f]

Simpson's rule is exact for polynomials up to degree 3.

---

## Gaussian Quadrature

Newton-Cotes fixes the nodes (equally spaced) and optimizes the weights.
Gaussian quadrature optimizes *both* nodes and weights to maximize the polynomial degree that is integrated exactly.

With \f$p\f$ points, Gauss-Legendre quadrature is exact for polynomials up to degree \f$2p - 1\f$:

\f[\int_{-1}^{1} f(x)\, dx \approx \sum_{i=1}^{p} w_i f(x_i)\f]

where \f$x_i\f$ are the roots of the \f$p\f$-th Legendre polynomial \f$P_p(x)\f$ and \f$w_i\f$ are precomputed weights.

For a general interval \f$[a, b]\f$, change of variables \f$x = \frac{a+b}{2} + \frac{b-a}{2} t\f$:

\f[\int_a^b f(x)\, dx = \frac{b-a}{2} \sum_{i=1}^{p} w_i f\!\left(\frac{a+b}{2} + \frac{b-a}{2} x_i\right)\f]

**Key property**: \f$p\f$ points achieve the accuracy of a degree-\f$(2p-1)\f$ polynomial fit, whereas \f$p\f$-point trapezoidal needs \f$O(p^2)\f$ points for the same accuracy on smooth functions.

The nodes and weights for \f$p = 1, \ldots, 5\f$ are hardcoded in the implementation (Abramowitz & Stegun, Table 25.4).

---

## Adaptive Quadrature

Fixed-panel methods waste evaluations in smooth regions and under-resolve oscillatory regions.
Adaptive methods refine only where the error estimate is large.

**Adaptive Simpson**: Estimate error on \f$[a, b]\f$ by comparing the single-panel estimate to the sum of two half-panel estimates. If \f$|\delta| > 15\,\varepsilon\f$, recurse on each half:

\f[\text{error} \approx \frac{S(a, b) - S(a, m) - S(m, b)}{15}\f]

where \f$m = (a+b)/2\f$. The factor \f$1/15\f$ comes from Richardson extrapolation applied to the \f$O(h^4)\f$ error of Simpson's rule.

**Properties**: Automatically concentrates evaluations near singularities and rapid variation. Total evaluations scale with the regularity of \f$f\f$, not the interval length.

---

## Romberg Integration

Romberg applies Richardson extrapolation to the trapezoidal rule. Let \f$T(h)\f$ denote the trapezoidal estimate with step size \f$h\f$. The Euler-Maclaurin expansion gives:

\f[T(h) = I + c_2 h^2 + c_4 h^4 + c_6 h^6 + \cdots\f]

The leading error term \f$c_2 h^2\f$ can be eliminated by combining estimates at \f$h\f$ and \f$h/2\f$:

\f[R_{i,j} = \frac{4^j R_{i,j-1} - R_{i-1,j-1}}{4^j - 1}\f]

This builds a triangular table:

\f[\begin{array}{cccc}
T(h) & & & \\
T(h/2) & R_{1,1} & & \\
T(h/4) & R_{2,1} & R_{2,2} & \\
T(h/8) & R_{3,1} & R_{3,2} & R_{3,3}
\end{array}\f]

The diagonal \f$R_{k,k}\f$ converges much faster than \f$T(h/2^k)\f$ alone. For smooth \f$f\f$, convergence is faster than any fixed power of \f$h\f$.

---

## Comparison

| Method | Error order | Function evaluations | Best for |
|---|---|---|---|
| Trapezoidal | \f$O(h^2)\f$ | \f$n+1\f$ | Baseline, periodic functions |
| Simpson | \f$O(h^4)\f$ | \f$n+1\f$ | Smooth functions, low cost |
| Gauss-Legendre (\f$p\f$ pts) | Exact for degree \f$\leq 2p-1\f$ | \f$p\f$ | Smooth functions, few evaluations |
| Adaptive Simpson | Error-controlled | Automatic | Irregular or oscillatory \f$f\f$ |
| Romberg | Faster than any \f$O(h^k)\f$ | \f$O(2^k)\f$ | Very smooth \f$f\f$, high accuracy |

---

## Parallel Structure

Quadrature is one of the most naturally parallel problems in scientific computing.

### Domain decomposition

Partition \f$[a, b]\f$ into \f$p\f$ subintervals, one per thread/process. Each computes its local integral; results are summed:

\f[\int_a^b f(x)\, dx = \sum_{i=0}^{p-1} \int_{a_i}^{b_i} f(x)\, dx\f]

This is a **parallel reduction** -- a fundamental pattern in GPU and MPI programming.

### Adaptive load balancing

Adaptive quadrature creates a work queue of subintervals to refine. This maps naturally to a task-parallel model (OpenMP tasks, CUDA dynamic parallelism), though load balancing is non-trivial.

### Gauss-Legendre with many panels

Apply \f$p\f$-point GL to each of \f$n\f$ panels independently -- all \f$n\f$ panels are independent. A GPU kernel with one thread block per panel achieves near-perfect parallelism.

---

## Exercises

1. Show that the trapezoidal rule is exact for \f$f(x) = \sin(x)\f$ integrated over a full period \f$[0, 2\pi]\f$ regardless of \f$n\f$. Why?

2. Derive the weights \f$\frac{1}{3}, \frac{4}{3}, \frac{1}{3}\f$ (times \f$h\f$) for Simpson's rule by integrating the Lagrange interpolant through \f$(a, f(a))\f$, \f$(m, f(m))\f$, \f$(b, f(b))\f$.

3. How many 5-point Gauss-Legendre evaluations does it take to integrate \f$x^9\f$ exactly on \f$[0,1]\f$? Verify.

4. Romberg integration of \f$f(x) = e^x\f$ on \f$[0, 1]\f$ using only 3 levels of refinement. Build the full \f$R\f$ table by hand.

5. (Parallel) You need \f$\int_0^{100} f(x)\, dx\f$ where \f$f\f$ is expensive (\f$10\f$ ms per evaluation). You have 64 GPU threads. Design a parallel trapezoidal scheme and estimate the wall-clock time compared to serial.
