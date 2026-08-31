# Ordinary Differential Equations {#page_ode}

Explicit, adaptive, and symplectic integrators for initial value problems (IVPs).

---

## 1. First-Order Initial Value Problems

Integrates \f$\frac{d\mathbf{y}}{dt} = \mathbf{f}(t, \mathbf{y})\f$ with \f$\mathbf{y}(t_0) = \mathbf{y}_0\f$.

### Common Parameter and Result Types

```cpp
struct ODEParams {
    double t0{0.0};       // Initial time
    double tf{1.0};       // Final time
    double h{1e-2};       // Initial step size
    double rtol{1e-6};    // Relative tolerance (adaptive methods)
    double atol{1e-9};    // Absolute tolerance (adaptive methods)
    int max_steps{100000};// Maximum allowed steps
};

struct ODEResult {
    num::Vector y;        // State at final time tf
    double t_final{0.0};  // Reached time
    int steps_taken{0};   // Count of successful steps
    bool success{false};  // True if reached tf without error
};
```

---

## 2. Explicit Integrators

### num::ode_euler
First-order forward Euler: \f$\mathbf{y}_{n+1} = \mathbf{y}_n + h \mathbf{f}(t_n, \mathbf{y}_n)\f$.

```cpp
num::ODEResult ode_euler(num::ODERhsFn f, num::Vector y0, num::ODEParams params);
```

### num::ode_rk4
Classical 4th-order Runge–Kutta (\f$\mathcal{O}(h^4)\f$).

```cpp
num::ODEResult ode_rk4(num::ODERhsFn f, num::Vector y0, num::ODEParams params);
```

### num::ode_rk45
5th-order Dormand–Prince with embedded 4th-order adaptive step size control.

```cpp
num::ODEResult ode_rk45(num::ODERhsFn f, num::Vector y0, num::ODEParams params,
                        num::ObserverFn observer = nullptr);
```

### Usage
```cpp
#include <numerics.hpp>

// Harmonic oscillator: y' = [v, -q]
auto oscillator = [](double, const num::Vector& y, num::Vector& dy) {
    dy[0] = y[1];
    dy[1] = -y[0];
};

num::ODEParams params{.t0 = 0.0, .tf = 20.0, .h = 0.01, .rtol = 1e-8, .atol = 1e-10};

// 1. Fixed-step RK4
num::ODEResult r_rk4 = num::ode_rk4(oscillator, {1.0, 0.0}, params);

// 2. Adaptive RK45 with step observer
auto observe = [](double t, const num::Vector& y) {
    // Invoked after each accepted step
};
num::ODEResult r_rk45 = num::ode_rk45(oscillator, {1.0, 0.0}, params, observe);
```

---

## 3. Second-Order Symplectic Systems

For separable Hamiltonian systems \f$\frac{d^2\mathbf{q}}{dt^2} = \mathbf{a}(\mathbf{q})\f$, preserving phase-space volume and energy invariants.

### num::ode_verlet
2nd-order Velocity Verlet:
\f[
\mathbf{v}_{n+1/2} = \mathbf{v}_n + \frac{h}{2}\mathbf{a}(\mathbf{q}_n), \quad
\mathbf{q}_{n+1} = \mathbf{q}_n + h\,\mathbf{v}_{n+1/2}, \quad
\mathbf{v}_{n+1} = \mathbf{v}_{n+1/2} + \frac{h}{2}\mathbf{a}(\mathbf{q}_{n+1})
\f]

```cpp
auto gravity = [](const num::Vector& q, num::Vector& a) {
    a[0] = -q[0];
    a[1] = -q[1];
};

num::Vector q0{1.0, 0.0}, v0{0.0, 1.0};
auto res = num::ode_verlet(gravity, q0, v0, params);
```

### num::ode_yoshida4
4th-order symplectic integrator via symmetric composition of Verlet substeps.

```cpp
auto res = num::ode_yoshida4(gravity, q0, v0, params);
```

### Lazy Trajectory Range (num::verlet)
Streams integration steps lazily without allocating trajectory arrays:

```cpp
for (const num::SymplecticStep& step : num::verlet(gravity, q0, v0, params)) {
    // step.t, step.q, step.v
}
```

---

## 4. Implicit Method of Lines (ode::advance)

Advances parabolic PDE fields via implicit backward Euler solves \f$(I - \Delta t \mathcal{L}) u^{n+1} = u^n\f$:

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

