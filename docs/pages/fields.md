# Field Examples {#page_fields}

## Two-Dimensional Grid

```cpp
#include <numerics.hpp>

num::grid2d grid{64, 1.0 / 65.0}; // 64 interior nodes per axis.

double x = grid.x(3);       // Physical x-coordinate.
double y = grid.y(5);       // Physical y-coordinate.
int flat = grid.flat(3, 5); // Row-major storage index.
int count = grid.size();    // 64 * 64 nodes.
```

## Two-Dimensional Scalar Field

```cpp
num::scalar_field_2d field(grid); // Allocate zero-filled contiguous storage.
field(3, 5) = 2.0;              // Access by grid coordinates.
```

## Sampling a Function

```cpp
#include <numbers>

num::scalar_field_2d field(grid, [](double x, double y) {
    return std::sin(std::numbers::pi * x) * std::sin(std::numbers::pi * y);
});
```

```cpp
field.fill([](double x, double y) {
    return x + y; // Replace values using physical coordinates.
});
```

## Field Storage

```cpp
num::vec& values = field.as_vec(); // Mutable view of field-owned storage.
const double* data = std::as_const(field).data();
num::idx size = field.size();
```

Changes through `values` are visible through `field(i, j)`.

## Three-Dimensional Grid

```cpp
num::grid_3d grid{32, 24, 16, 0.1, -1.0, 0.0, 2.0};

num::idx flat = grid.flat(3, 5, 7); // x varies fastest in storage.
double x = grid.x(3);               // Origin plus i * dx.
int count = grid.size();
```

## Three-Dimensional Scalar Field

```cpp
num::scalar_field_3d density(32, 24, 16, 0.1f); // Zero-filled field at the origin.
density(3, 5, 7) = 4.0;
density.set(3, 5, 7, 4.0);                    // Equivalent explicit setter.
```

## Field Origin

```cpp
num::scalar_field_3d density(
    32, 24, 16, 0.1f, -1.0f, 0.0f, 2.0f); // Set physical origin.
```

## Filling Three-Dimensional Fields

```cpp
density.fill(1.0); // Replace every value with a constant.
```

```cpp
density.fill([](int i, int j, int k) {
    return static_cast<double>(i + j + k); // Callable receives integer indices.
});
```

## Trilinear Sampling

```cpp
float value = density.sample(0.15f, 0.25f, 0.35f); // Interpolate physical coordinates.
```

## Three-Dimensional Vector Field

```cpp
num::vector_field_3d velocity(32, 24, 16, 0.1f);
velocity.x.fill(1.0);
velocity.y.fill(0.0);
velocity.z.fill(-1.0);
```

```cpp
std::array<float, 3> value = velocity.sample(0.15f, 0.25f, 0.35f);
velocity.scale(0.5f); // Scale all three component fields.
```

## Gradient (\f$\nabla \phi\f$)

\f[
\nabla \phi = \left( \frac{\partial \phi}{\partial x}, \frac{\partial \phi}{\partial y}, \frac{\partial \phi}{\partial z} \right)
\f]

```cpp
num::vector_field_3d gradient = num::field_solver::gradient(density);
```

## Divergence (\f$\nabla \cdot \mathbf{v}\f$)

\f[
\nabla \cdot \mathbf{v} = \frac{\partial v_x}{\partial x} + \frac{\partial v_y}{\partial y} + \frac{\partial v_z}{\partial z}
\f]

```cpp
num::scalar_field_3d divergence = num::field_solver::divergence(velocity);
```

## Curl (\f$\nabla \times \mathbf{v}\f$)

\f[
\nabla \times \mathbf{v} = \left( \frac{\partial v_z}{\partial y} - \frac{\partial v_y}{\partial z},\; \frac{\partial v_x}{\partial z} - \frac{\partial v_z}{\partial x},\; \frac{\partial v_y}{\partial x} - \frac{\partial v_x}{\partial y} \right)
\f]

```cpp
num::vector_field_3d curl = num::field_solver::curl(velocity);
```

## 3D Poisson Solve (\f$-\nabla^2 \phi = \rho\f$)

```cpp
num::scalar_field_3d potential(32, 24, 16, 0.1f);
num::solver_result result =
    num::field_solver::solve_poisson(potential, density, 1e-8, 1000);
```

The source and destination must share the same grid geometry.

## Variable-Coefficient Poisson Solve (\f$-\nabla \cdot (\sigma \nabla \phi) = \rho\f$)

```cpp
num::scalar_field_3d coefficient(32, 24, 16, 0.1f);
coefficient.fill(1.0);

num::array<num::field_solver::dirichlet_bc> boundaries{
    {static_cast<int>(coefficient.grid().flat(0, 0, 0)), 1.0},
};

auto result = num::field_solver::solve_var_poisson(
    potential, coefficient, boundaries, 1e-8, 1000);
```

## Current Density (\f$\mathbf{J} = -\sigma \nabla \phi\f$)

```cpp
num::vector_field_3d current =
    num::magnetic_solver::current_density(coefficient, potential); // J = -sigma * grad(phi)
```

## Magnetic Field (\f$\nabla^2 \mathbf{A} = -\mu_0 \mathbf{J}, \quad \mathbf{B} = \nabla \times \mathbf{A}\f$)

```cpp
num::vector_field_3d current(16, 16, 16, 0.1f);   // the source current density J

num::vector_field_3d magnetic =
    num::magnetic_solver::solve_magnetic_field(current, 1e-8, 1000);
```

The 3D Poisson program is shown on @ref page_pde.
