# ODE Module {#page_ode}

`include/ode/ode.hpp` -- ordinary differential equation integrators.

---

## Types

```cpp
namespace num {

using ODERhsFn = std::function<void(real t, const Vector& y, Vector& dy)>;
using AccelFn  = std::function<void(const Vector& q, Vector& acc)>;

struct Step          { real t; Vector y; };
struct SymplecticStep{ real t; Vector q; Vector v; };

struct ODEResult     { Vector y; real t; idx steps; bool converged; };
struct SymplecticResult { Vector q, v; real t; idx steps; };

} // namespace num
```

---

## Usage patterns

### Final state only

```cpp
auto res = ode_rk4(f, y0, 0.0, 10.0, 0.01);
// res.y  -- final state
// res.t  -- final time
// res.steps -- steps taken
```

### Observe every step

Each integrator has a lazy-range factory (`rk4`, `rk45`, `verlet`, etc.) that
yields one `Step` or `SymplecticStep` per iteration. Use a range-based for:

```cpp
// Each factory returns a lazy range; each iteration yields one Step {t, y} or SymplecticStep {t, q, v}.
std::vector<real> ts, xs;
for (auto [t, y] : num::rk4(f, y0, 0.0, 10.0, 0.01)) {
    ts.push_back(t);
    xs.push_back(y[0]);
}

real max_drift = 0.0;
real E0 = energy(q0, v0);
for (auto [t, q, v] : num::verlet(accel, q0, v0, 0.0, 1000.0, 0.01))
    max_drift = std::max(max_drift, std::abs(energy(q, v) - E0));
```

---

## Integrators

### Explicit Euler -- O(h)

```cpp
// High-level
ODEResult ode_euler(ODERhsFn f, Vector y0, real t0, real t1, real h);

// Lazy range
EulerSteps euler(ODERhsFn f, Vector y0, real t0, real t1, real h);
```

Fixed-step forward Euler. One function evaluation per step.

\f[y_{n+1} = y_n + h\,f(t_n,\,y_n)\f]

Global error \f$O(h)\f$.

### Classic RK4 -- O(h^4)

```cpp
ODEResult ode_rk4(ODERhsFn f, Vector y0, real t0, real t1, real h);
RK4Steps  rk4    (ODERhsFn f, Vector y0, real t0, real t1, real h);
```

Fixed-step 4th-order Runge-Kutta. Four evaluations of `f` per step.

\f[k_1 = f(t_n, y_n), \quad k_2 = f\!\left(t_n + \tfrac{h}{2},\, y_n + \tfrac{h}{2}k_1\right)\f]
\f[k_3 = f\!\left(t_n + \tfrac{h}{2},\, y_n + \tfrac{h}{2}k_2\right), \quad k_4 = f(t_n + h,\, y_n + h k_3)\f]
\f[y_{n+1} = y_n + \tfrac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)\f]

Good default for smooth, non-Hamiltonian problems.

### Adaptive RK45 -- O(h^5) with error control

```cpp
ODEResult ode_rk45(ODERhsFn f, Vector y0, real t0, real t1,
                   real rtol = 1e-6, real atol = 1e-9,
                   real h0 = 1e-3, idx max_steps = 1000000);

RK45Steps rk45    (ODERhsFn f, Vector y0, real t0, real t1,
                   real rtol = 1e-6, real atol = 1e-9,
                   real h0 = 1e-3, idx max_steps = 1000000);
```

Dormand-Prince embedded 4(5) pair with FSAL (6 evaluations per accepted step).
Step accepted when the mixed error norm satisfies

\f[\left\|\frac{e_i}{\text{atol} + |y_i|\,\text{rtol}}\right\|_\infty \leq 1\f]

`res.converged` is `false` if `max_steps` was hit before reaching `t1`.

### Velocity Verlet -- O(h^2), symplectic

```cpp
SymplecticResult ode_verlet(AccelFn accel, Vector q0, Vector v0,
                             real t0, real t1, real h);

VerletSteps      verlet    (AccelFn accel, Vector q0, Vector v0,
                             real t0, real t1, real h);
```

Kick-drift-kick velocity Verlet with FSAL acceleration caching.
Energy error oscillates within a bounded band rather than growing secularly.

\f[q_{n+1} = q_n + h\,v_n + \tfrac{h^2}{2}\,a_n\f]
\f[v_{n+1} = v_n + \tfrac{h}{2}(a_n + a_{n+1})\f]

### Yoshida 4th-order -- O(h^4), symplectic

```cpp
SymplecticResult ode_yoshida4(AccelFn accel, Vector q0, Vector v0,
                               real t0, real t1, real h);

Yoshida4Steps    yoshida4   (AccelFn accel, Vector q0, Vector v0,
                               real t0, real t1, real h);
```

Yoshida (1990) 4th-order symplectic integrator. Three force evaluations per step
using a drift-kick sequence with coefficients

\f[w_1 = \frac{1}{2 - 2^{1/3}}, \quad w_0 = 1 - 2w_1\f]

### RK4 for second-order systems (Nystrom form)

```cpp
SymplecticResult ode_rk4_2nd(AccelFn accel, Vector q0, Vector v0,
                              real t0, real t1, real h);

RK4_2ndSteps     rk4_2nd    (AccelFn accel, Vector q0, Vector v0,
                              real t0, real t1, real h);
```

Standard RK4 in Nystrom form. Four force evaluations per step; **not symplectic**
(energy drifts secularly). Use `ode_verlet` or `ode_yoshida4` for long Hamiltonian runs.

---

## Symplectic vs. RK4: Energy Conservation

| Property | Verlet \f$O(h^2)\f$ | Yoshida4 \f$O(h^4)\f$ | RK4 \f$O(h^4)\f$ |
|----------|----------------|------------------|-------------|
| Symplectic | Yes | Yes | No |
| Energy error | Bounded oscillation | Bounded oscillation | Secular drift |
| Force evals/step | 1 (FSAL) | 3 | 4 |

---

## Example: Lorenz attractor

```cpp
#include "ode/ode.hpp"
using namespace num;

auto lorenz = [](double, const Vector& s, Vector& ds) {
    ds[0] = 10.0 * (s[1] - s[0]);
    ds[1] = s[0] * (28.0 - s[2]) - s[1];
    ds[2] = s[0] * s[1] - (8.0/3.0) * s[2];
};

Series xz;
// Runge-Kutta RK4(5): t=[0, 50], rtol=1e-8, atol=1e-10 — each iteration is one accepted Step {t, y}.
for (auto [t, y] : rk45(lorenz, {1.0, 0.0, 0.0}, 0.0, 50.0, 1e-8, 1e-10))
    xz.emplace_back(y[0], y[2]);
```

## Example: Kepler orbit

```cpp
AccelFn accel = [](const Vector& q, Vector& a) {
    real r3 = std::pow(q[0]*q[0] + q[1]*q[1], 1.5);
    a[0] = -q[0] / r3;  a[1] = -q[1] / r3;
};

auto res = ode_yoshida4(accel, {1.0, 0.0}, {0.0, 1.0}, 0.0, 2.0*M_PI, 0.01);
// res.q ~= {1, 0}, res.v ~= {0, 1}
```

---

## Tests

`tests/test_ode.cpp` covers:

- **ODE_Euler**: convergence order ~= 1.0 on y' = -y
- **ODE_RK4**: convergence order ~= 4.0 on harmonic oscillator; full-period accuracy
- **ODE_RK45**: adaptive step on harmonic oscillator and exponential decay
- **ODE_Stepper**: trajectory recording via range-based for; RK45 step count
- **ODE_Verlet**: circular orbit return; bounded energy over t = 100; better than RK4 over t = 200; step count
- **ODE_Yoshida4**: circular orbit; higher-order accuracy vs. Verlet; bounded energy < 1e-8

---

## References

- E. Hairer, C. Lubich, G. Wanner, *Geometric Numerical Integration*, 2nd ed., Springer (2006)
- H. Yoshida, *Construction of higher order symplectic integrators*, Phys. Lett. A **150** (1990) 262-268
- J. R. Dormand & P. J. Prince, *A family of embedded Runge-Kutta formulae*, J. Comput. Appl. Math. **6** (1980)
