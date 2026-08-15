# Application Example: Projection Step {#page_ns_perf}

This page shows the numerics pattern used by a matrix-free pressure projection.

## Matrix-Free Pressure Operator

```cpp
auto A = num::operators::make_op(
    [&](const num::Vector& p, num::Vector& Ap) {
        apply_negative_laplacian_with_boundary_rows(p, Ap, nx, ny);
    },
    nx * ny);
```

## Solve the Poisson System

```cpp
num::Vector pressure(nx * ny, 0.0);
num::SolverResult info =
    num::cg(num::operators::assume_spd(A), rhs, pressure, 1e-8, 1000, num::Backend::blas);
```

The same code can switch to the sequential backend for diagnostics:

```cpp
num::SolverResult ref =
    num::cg(num::operators::assume_spd(A), rhs, pressure, 1e-10, 2000, num::Backend::seq);
```

## Projection Skeleton

```cpp
advect_velocity(u, v, dt);
build_divergence_rhs(u, v, rhs);

num::cg(num::operators::assume_spd(A), rhs, pressure, 1e-8, 1000, num::best_backend);

subtract_pressure_gradient(u, v, pressure, dt);
apply_boundary_conditions(u, v);
```

This example is intentionally matrix-free: the pressure matrix is represented by
the stencil application, not by assembled CSR storage.
