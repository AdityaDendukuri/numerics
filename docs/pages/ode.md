# Ordinary Differential Equations {#page_ode}

The `ode` module provides explicit, adaptive, and symplectic integrators for initial value problems (IVPs).

---

## 1. First-Order Initial Value Problems

Integrates systems of first-order differential equations:

\f[
\frac{d\mathbf{y}}{dt} = \mathbf{f}(t, \mathbf{y}), \qquad \mathbf{y}(t_0) = \mathbf{y}_0
\f]

```cpp
#include <numerics.hpp>

// Harmonic oscillator: q' = v, v' = -q
num::ODERhsFn oscillator = [](double, const num::Vector& y, num::Vector& dy) {
    dy[0] = y[1];  // dq/dt = v
    dy[1] = -y[0]; // dv/dt = -q
};
```

---

## 2. Integration Parameters & Solvers

```cpp
num::ODEParams params{
    .t0 = 0.0,
    .tf = 20.0,
    .h = 1e-2,
    .rtol = 1e-8,
    .atol = 1e-10,
};
```

### Forward Euler (\f$\mathcal{O}(h)\f$)

\f[
\mathbf{y}_{n+1} = \mathbf{y}_n + h\,\mathbf{f}(t_n, \mathbf{y}_n)
\f]

```cpp
num::ODEResult result = num::ode_euler(oscillator, {1.0, 0.0}, params);
```

### Classical Runge–Kutta 4th-Order (\f$\mathcal{O}(h^4)\f$)

\f[
\begin{aligned}
\mathbf{k}_1 &= \mathbf{f}(t_n, \mathbf{y}_n), \\
\mathbf{k}_2 &= \mathbf{f}\left(t_n + \tfrac{h}{2}, \mathbf{y}_n + \tfrac{h}{2}\mathbf{k}_1\right), \\
\mathbf{k}_3 &= \mathbf{f}\left(t_n + \tfrac{h}{2}, \mathbf{y}_n + \tfrac{h}{2}\mathbf{k}_2\right), \\
\mathbf{k}_4 &= \mathbf{f}(t_n + h, \mathbf{y}_n + h\mathbf{k}_3), \\
\mathbf{y}_{n+1} &= \mathbf{y}_n + \frac{h}{6}\left(\mathbf{k}_1 + 2\mathbf{k}_2 + 2\mathbf{k}_3 + \mathbf{k}_4\right)
\end{aligned}
\f]

```cpp
num::ODEResult result = num::ode_rk4(oscillator, {1.0, 0.0}, params);
```

### Adaptive Dormand–Prince RK45

Computes 5th-order accurate trajectories with embedded 4th-order error estimation:

\f[
e_n = \|\mathbf{y}^{(5)}_{n+1} - \mathbf{y}^{(4)}_{n+1}\|_\infty, \qquad h_{\text{new}} = h \cdot \min\left(2.0, \max\left(0.2, 0.9 \left(\frac{\text{tol}}{e_n}\right)^{1/5}\right)\right)
\f]

```cpp
num::ODEResult result = num::ode_rk45(oscillator, {1.0, 0.0}, params);
```

### Step Observers & Telemetry

```cpp
num::ObserverFn observe = [](double t, const num::Vector& state) {
    record(t, state); // Invoked after each accepted step
};

num::ODEResult result = num::ode_rk45(oscillator, {1.0, 0.0}, params, observe);
```

---

## 3. Second-Order Hamiltonian & Symplectic Systems

For separable mechanical systems governed by accelerations \f$\mathbf{a}(\mathbf{q})\f$:

\f[
\frac{d^2\mathbf{q}}{dt^2} = \mathbf{a}(\mathbf{q}), \qquad \frac{d\mathbf{q}}{dt} = \mathbf{v}, \quad \frac{d\mathbf{v}}{dt} = \mathbf{a}(\mathbf{q})
\f]

Symplectic integrators preserve phase-space volume (\f$d\mathbf{p} \wedge d\mathbf{q}\f$) and prevent secular energy drift over astronomical time horizons.

```cpp
num::AccelFn gravity = [](const num::Vector& q, num::Vector& acceleration) {
    acceleration[0] = -q[0];
    acceleration[1] = -q[1];
};

num::Vector q0{1.0, 0.0};
num::Vector v0{0.0, 1.0};
```

### Velocity Verlet (2nd-Order Symplectic)

\f[
\mathbf{v}_{n+1/2} = \mathbf{v}_n + \frac{h}{2}\mathbf{a}(\mathbf{q}_n), \qquad \mathbf{q}_{n+1} = \mathbf{q}_n + h\,\mathbf{v}_{n+1/2}, \qquad \mathbf{v}_{n+1} = \mathbf{v}_{n+1/2} + \frac{h}{2}\mathbf{a}(\mathbf{q}_{n+1})
\f]

```cpp
auto result = num::ode_verlet(gravity, q0, v0, params); // 2nd-order symplectic Verlet
```

### Yoshida 4th-Order Symplectic

Constructed by symmetric composition of three Verlet substeps with weights \f$w_1 = w_3 = \frac{1}{2 - 2^{1/3}}\f$, \f$w_2 = -\frac{2^{1/3}}{2 - 2^{1/3}}\f$:

```cpp
auto result = num::ode_yoshida4(gravity, q0, v0, params); // 4th-order symplectic energy conservation
```

---

## 4. Lazy Range Trajectories

Iterators provide memory-efficient step streaming without materializing the full orbit array in RAM:

```cpp
for (const num::SymplecticStep& step : num::verlet(gravity, q0, v0, params)) {
    record_orbit(step.t, step.q, step.v);
}
```

---

## 5. Implicit PDE Steppers (Method of Lines)

Advances parabolic PDE fields \f$\frac{\partial u}{\partial t} = \mathcal{L}(u)\f$ via implicit Backward Euler solves \f$(I - \Delta t \mathcal{L}) u^{n+1} = u^n\f$:

```cpp
num::operators::SparseOp op(A);
auto spd = num::operators::assume_spd(op);

num::LinearSolver solver = [&](const num::Vector& rhs, num::Vector& x) {
    return num::cg(spd, rhs, x, 1e-8, 1000);
};

num::ode::advance(field, solver, {.nstep = 100, .dt = 1e-3});
```

---

## Complete Example

@example 05_symplectic_nbody_ode.cpp
