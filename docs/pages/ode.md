# ODE Module {#page_ode}

`include/ode/ode.hpp` -- ordinary differential equation integrators with optional step callbacks.

---

## Types

```cpp
namespace num {

// State-space RHS: dy/dt = f(t, y)
using ODERhsFn = std::function<void(real t, const Vector& y, Vector& dy)>;

// Called after each accepted step (state-space integrators)
using StepCallback = std::function<void(real t, const Vector& y)>;

// Gravitational / Hamiltonian acceleration: a = a(q)
using AccelFn = std::function<void(const Vector& q, Vector& acc)>;

// Called after each step (symplectic integrators)
using SymplecticCallback = std::function<void(real t, const Vector& q, const Vector& v)>;

// Return type for state-space integrators
struct ODEResult { Vector y; real t; idx steps; bool converged; };

// Return type for symplectic integrators
struct SymplecticResult { Vector q, v; real t; idx steps; };

} // namespace num
```

---

## Integrators

### Explicit Euler -- O(h)

```cpp
ODEResult ode_euler(ODERhsFn f, Vector y0,
                    real t0, real t1, real h,
                    StepCallback on_step = nullptr);
```

Fixed-step forward Euler. One function evaluation per step.

$$y_{n+1} = y_n + h\,f(t_n,\,y_n)$$

Global error $O(h)$. Useful as a baseline or for stiff systems where stability dominates.

### Classic RK4 -- O(h^4)

```cpp
ODEResult ode_rk4(ODERhsFn f, Vector y0,
                  real t0, real t1, real h,
                  StepCallback on_step = nullptr);
```

Fixed-step 4th-order Runge-Kutta. Four evaluations of `f` per step.

$$k_1 = f(t_n, y_n), \quad k_2 = f\!\left(t_n + \tfrac{h}{2},\, y_n + \tfrac{h}{2}k_1\right)$$
$$k_3 = f\!\left(t_n + \tfrac{h}{2},\, y_n + \tfrac{h}{2}k_2\right), \quad k_4 = f(t_n + h,\, y_n + h k_3)$$
$$y_{n+1} = y_n + \tfrac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$

Good default for smooth, non-Hamiltonian problems.

### Adaptive RK45 -- O(h^5) with error control

```cpp
ODEResult ode_rk45(ODERhsFn f, Vector y0,
                   real t0, real t1,
                   real rtol = 1e-6, real atol = 1e-9,
                   real h0 = 1e-3, idx max_steps = 1000000,
                   StepCallback on_step = nullptr);
```

Dormand-Prince embedded 4(5) pair. Step accepted when the mixed error norm satisfies

$$\left\|\frac{e_i}{\text{atol} + |y_i|\,\text{rtol}}\right\|_\infty \leq 1$$

Step size updated via PI control:

$$h_\text{new} = h \cdot \min\!\left(10,\, \max\!\left(0.1,\; 0.9\,\|\varepsilon\|^{-0.2}\right)\right)$$

`res.converged` is `false` if `max_steps` was hit before reaching `t1`.

### Velocity Verlet -- O(h^2), symplectic

```cpp
SymplecticResult ode_verlet(AccelFn accel,
                             Vector q0, Vector v0,
                             real t0, real t1, real h,
                             SymplecticCallback on_step = nullptr);
```

Kick-drift-kick velocity Verlet. FSAL: one force evaluation per step after initialisation.
Preserves the symplectic structure of Hamilton's equations; energy error oscillates within a bounded band rather than growing secularly.

$$q_{n+1} = q_n + h\,v_n + \tfrac{h^2}{2}\,a_n$$
$$a_{n+1} = \text{accel}(q_{n+1})$$
$$v_{n+1} = v_n + \tfrac{h}{2}(a_n + a_{n+1})$$

### Yoshida 4th-order -- O(h^4), symplectic

```cpp
SymplecticResult ode_yoshida4(AccelFn accel,
                               Vector q0, Vector v0,
                               real t0, real t1, real h,
                               SymplecticCallback on_step = nullptr);
```

Yoshida (1990) 4th-order symplectic integrator. Three force evaluations per step.
Built from a drift-kick sequence using coefficients

$$w_1 = \frac{1}{2 - 2^{1/3}}, \quad w_0 = 1 - 2w_1$$
$$c_1 = c_4 = \tfrac{w_1}{2}, \quad c_2 = c_3 = \tfrac{w_0 + w_1}{2}, \quad d_1 = d_3 = w_1, \quad d_2 = w_0$$

One step executes: drift($c_1$) -- kick($d_1$) -- drift($c_2$) -- kick($d_2$) -- drift($c_2$) -- kick($d_1$) -- drift($c_1$).

### RK4 for second-order systems (Nystrom form)

```cpp
SymplecticResult ode_rk4_2nd(AccelFn accel,
                              Vector q0, Vector v0,
                              real t0, real t1, real h,
                              SymplecticCallback on_step = nullptr);
```

Standard RK4 applied to $[q,v]' = [v,\, \text{accel}(q)]$, kept in Nystrom form to avoid packing/unpacking. Four force evaluations per step; global error $O(h^4)$ but **not symplectic** (energy drifts secularly).

$$a_1 = \text{accel}(q), \quad a_2 = \text{accel}\!\left(q + \tfrac{h}{2}v\right)$$
$$a_3 = \text{accel}\!\left(q + \tfrac{h}{2}\!\left(v + \tfrac{h}{2}a_1\right)\right), \quad a_4 = \text{accel}\!\left(q + h\!\left(v + \tfrac{h}{2}a_2\right)\right)$$
$$q_{n+1} = q_n + h\,v_n + \tfrac{h^2}{6}(a_1 + a_2 + a_3)$$
$$v_{n+1} = v_n + \tfrac{h}{6}(a_1 + 2a_2 + 2a_3 + a_4)$$

---

## Step Callbacks

Both state-space and symplectic integrators accept an optional callback fired after each accepted step. This enables trajectory recording without the integrator allocating storage:

```cpp
std::vector<real> ts, xs;
ode_rk4(f, y0, 0.0, 10.0, 0.01,
        [&](real t, const Vector& y) {
            ts.push_back(t);
            xs.push_back(y[0]);
        });
```

```cpp
real max_drift = 0.0;
real E0 = energy(q0, v0);
ode_verlet(accel, q0, v0, 0.0, 1000.0, 0.01,
           [&](real, const Vector& q, const Vector& v) {
               max_drift = std::max(max_drift,
                                    std::abs(energy(q,v) - E0));
           });
```

---

## Symplectic vs. RK4: Energy Conservation

For Hamiltonian systems (orbital mechanics, molecular dynamics), symplectic integrators are strongly preferred:

| Property | Verlet $O(h^2)$ | Yoshida4 $O(h^4)$ | RK4 $O(h^4)$ |
|----------|----------------|------------------|-------------|
| Symplectic | Yes | Yes | No |
| Energy error | Bounded oscillation | Bounded oscillation | Secular drift |
| Force evals/step | 1 (FSAL) | 3 | 4 |

The `apps/nbody` demo visualises this directly: cycle between Verlet, Yoshida4, and RK4 on a Kepler orbit and watch the energy panel.

---

## Example: Kepler Orbit

```cpp
#include "ode/ode.hpp"
using namespace num;

// Circular orbit at r=1: q=(1,0), v=(0,1), exact period T=2*pi
AccelFn accel = [](const Vector& q, Vector& a) {
    real r3 = std::pow(q[0]*q[0] + q[1]*q[1], 1.5);
    a[0] = -q[0] / r3;
    a[1] = -q[1] / r3;
};

auto res = ode_yoshida4(accel, {1.0, 0.0}, {0.0, 1.0},
                         0.0, 2.0 * M_PI, 0.01);
// res.q ~= {1, 0}, res.v ~= {0, 1}  (returns to start after one orbit)
```

---

## Tests

`tests/test_ode.cpp` covers:

- **ODE_Euler**: convergence order ~= 1.0 on y' = -y
- **ODE_RK4**: convergence order ~= 4.0 on harmonic oscillator; full-period accuracy
- **ODE_RK45**: adaptive step on harmonic oscillator and exponential decay
- **ODE_Callback**: trajectory recording (count, last time, last value)
- **ODE_Verlet**: circular orbit return; bounded energy over t = 100; better energy conservation than RK4 over t = 200
- **ODE_Yoshida4**: circular orbit; higher-order accuracy vs. Verlet; bounded energy < 1e-8

---

## References

- E. Hairer, C. Lubich, G. Wanner, *Geometric Numerical Integration*, 2nd ed., Springer (2006)
- H. Yoshida, *Construction of higher order symplectic integrators*, Phys. Lett. A **150** (1990) 262-268
- J. R. Dormand & P. J. Prince, *A family of embedded Runge-Kutta formulae*, J. Comput. Appl. Math. **6** (1980)
