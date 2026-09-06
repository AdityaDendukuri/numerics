# Application Example: Projection Step {#page_ns_perf}

This page shows the numerics pattern used by a matrix-free pressure projection.

## Matrix-Free Pressure Operator

```cpp
auto A = num::operators::make_op(
    [&](const num::vec& p, num::vec& Ap) {
        apply_negative_laplacian_with_boundary_rows(p, Ap, nx, ny);
    },
    nx * ny);
```

## Solve the Poisson System

```cpp
num::vec pressure(nx * ny, 0.0);
num::solver_result info =
    num::cg(num::operators::assume_spd(A), rhs, pressure, 1e-8, 1000);
```

`num::cg` always runs the level-1 work through `num::accel` (see @ref
page_parallel), so this already picks up BLAS/OMP if the build has them. There
is no sequential override at the `cg` call site — to compare against the
portable reference directly, call the vector ops it's built from
(`num::seq::dot`, `num::seq::axpy`) instead of `num::cg` for diagnostics.

## Projection Skeleton

```cpp
advect_velocity(u, v, dt);
build_divergence_rhs(u, v, rhs);

num::cg(num::operators::assume_spd(A), rhs, pressure, 1e-8, 1000);

subtract_pressure_gradient(u, v, pressure, dt);
apply_boundary_conditions(u, v);
```

The pressure matrix is represented by the stencil application, not by
assembled CSR storage.
