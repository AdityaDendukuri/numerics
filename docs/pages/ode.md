# ODE Examples {#page_ode}

## First-Order System

```cpp
#include <numerics.hpp>

num::ODERhsFn oscillator = [](double, const num::Vector& y, num::Vector& dy) {
    dy[0] = y[1];  // q' = v.
    dy[1] = -y[0]; // v' = -q.
};
```

## Integration Parameters

```cpp
num::ODEParams params{
    .t0 = 0.0,
    .tf = 20.0,
    .h = 1e-2,
    .rtol = 1e-8,
    .atol = 1e-10,
};
```

Fixed-step solvers use `t0`, `tf`, and `h`. RK45 also uses the tolerances and
`max_steps`.

## Forward Euler

```cpp
num::ODEResult result = num::ode_euler(oscillator, {1.0, 0.0}, params);
```

## Classical RK4

```cpp
num::ODEResult result = num::ode_rk4(oscillator, {1.0, 0.0}, params);
```

## Adaptive RK45

```cpp
num::ODEResult result = num::ode_rk45(oscillator, {1.0, 0.0}, params);
```

## Observing Accepted Steps

```cpp
num::ObserverFn observe = [](double t, const num::Vector& state) {
    record(t, state); // Called after each accepted step.
};

num::ODEResult result = num::ode_rk45(oscillator, {1.0, 0.0}, params, observe);
```

## Result Metadata

```cpp
num::Vector final_state = result.u; // Result owns the final state.
double final_time = result.t;
num::idx accepted = result.steps;
bool reached_end = result.converged;
```

## Problem and Algorithm Objects

```cpp
num::ODEProblem problem{oscillator, {1.0, 0.0}, 0.0, 20.0};
num::RK45 method{.h = 1e-2, .rtol = 1e-8, .atol = 1e-10};

num::ODEResult result = num::solve(problem, method);
```

```cpp
auto euler_result = num::solve(problem, num::Euler{.h = 1e-3});
auto rk4_result = num::solve(problem, num::RK4{.h = 1e-2});
```

## Lazy First-Order Steps

```cpp
auto trajectory = num::rk4(oscillator, {1.0, 0.0}, params);

for (const num::Step& step : trajectory) {
    record(step.t, step.u); // Each snapshot owns its state.
}
```

```cpp
auto trajectory = num::rk45(oscillator, {1.0, 0.0}, params);
num::ODEResult final = trajectory.run(); // Consume the range and keep only the result.
```

`num::euler`, `num::rk4`, and `num::rk45` provide the same lazy interface.

## Second-Order System

```cpp
num::AccelFn gravity = [](const num::Vector& q, num::Vector& acceleration) {
    acceleration[0] = -q[0];
    acceleration[1] = -q[1];
};

num::Vector q0{1.0, 0.0};
num::Vector v0{0.0, 1.0};
```

## Velocity Verlet

```cpp
auto result = num::ode_verlet(gravity, q0, v0, params); // Second-order symplectic update.
```

## Fourth-Order Yoshida

```cpp
auto result = num::ode_yoshida4(gravity, q0, v0, params); // Fourth-order symplectic update.
```

## Fourth-Order Nystrom RK

```cpp
auto result = num::ode_rk4_2nd(gravity, q0, v0, params); // Fourth-order non-symplectic update.
```

## Lazy Symplectic Steps

```cpp
for (const num::SymplecticStep& step : num::verlet(gravity, q0, v0, params)) {
    record_orbit(step.t, step.q, step.v);
}
```

`num::verlet`, `num::yoshida4`, and `num::rk4_2nd` provide lazy trajectories.

## Implicit Field Updates

```cpp
num::operators::SparseOp op(A);

num::LinearSolver solver = [&](const num::Vector& rhs, num::Vector& x) {
    return num::cg(num::operators::assume_spd(op), rhs, x, 1e-8, 1000);
};

num::ode::advance(field, solver, {.nstep = 100, .dt = 1e-3});
```

## Observing Implicit Updates

```cpp
num::ode::advance(field, solver, {.nstep = 100, .dt = 1e-3},
    [](int step, double time, const auto& current) {
        save(step, time, current); // Includes the initial field at step zero.
    });
```

Any field exposing `vec()` satisfies the implicit stepper interface.

## Complete Program

@example 05_symplectic_nbody_ode.cpp
