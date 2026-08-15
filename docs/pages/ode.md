# ODE Examples {#page_ode}

ODE routines cover fixed-step explicit schemes, adaptive RK45, symplectic
second-order schemes, and implicit field updates.

## Adaptive RK45

```cpp
#include <numerics.hpp>

auto rhs = [](double, const num::Vector& y, num::Vector& dy) {
    dy[0] = y[1];
    dy[1] = -y[0];
};

auto result = num::solve(
    num::ODEProblem{rhs, {1.0, 0.0}, 0.0, 20.0},
    num::RK45{.h = 1e-2, .rtol = 1e-8, .atol = 1e-10});
```

`result.u` is the final state, `result.t` is the final time, and
`result.steps` is the accepted step count.

## Fixed-Step RK4 With Samples

```cpp
auto steps = num::rk4(rhs, {1.0, 0.0}, {.t0 = 0.0, .tf = 10.0, .h = 1e-2});

for (const auto& s : steps) {
    const double energy = 0.5 * (s.u[0] * s.u[0] + s.u[1] * s.u[1]);
    (void)energy;
}
```

The range API is useful when every accepted step is part of the output.

## Symplectic Verlet

```cpp
auto accel = [](const num::Vector& q, num::Vector& a) {
    a[0] = -q[0];
    a[1] = -q[1];
};

num::Vector q0{1.0, 0.0};
num::Vector v0{0.0, 1.0};

auto orbit = num::ode_verlet(accel, q0, v0, {.t0 = 0.0, .tf = 100.0, .h = 1e-2});
```

Use `ode_yoshida4` when a fourth-order symplectic update is required.

## Backward Euler Field Step

```cpp
num::operators::SparseOp Aop(A);

num::LinearSolver solver = [&](const num::Vector& rhs, num::Vector& x) {
    return num::cg(num::operators::assume_spd(Aop), rhs, x, 1e-8, 1000).converged;
};

num::ode::advance(u, solver, {.nstep = nsteps, .dt = dt});
```

This form separates the implicit update from the linear solver used at each
step.
