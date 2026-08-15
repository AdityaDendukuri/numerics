# Field Examples {#page_fields}

Field types provide grid storage for PDE and vector-calculus examples.

## Scalar Field

```cpp
const int nx = 64;
const int ny = 64;
const int nz = 64;
const float dx = 1.0f / 63.0f;

num::ScalarField3D rho(nx, ny, nz, dx);
rho.fill([&](int i, int j, int k) {
    const double x = i * dx;
    const double y = j * dx;
    const double z = k * dx;
    return std::exp(-40.0 * ((x - 0.5) * (x - 0.5)
                           + (y - 0.5) * (y - 0.5)
                           + (z - 0.5) * (z - 0.5)));
});
```

## Vector Field

```cpp
num::VectorField3D v(nx, ny, nz, dx);
v.x.fill(0.0);
v.y.fill(0.0);
v.z.fill(1.0);
```

## Poisson Utility

```cpp
num::ScalarField3D phi(nx, ny, nz, dx);

num::SolverResult info =
    num::FieldSolver::solve_poisson(phi, rho, 1e-8, 1000);
```

Internally this wraps the finite-difference operator with
`num::operators::make_op` and solves with conjugate gradients.

## Magnetic Field Utility

```cpp
num::VectorField3D J(nx, ny, nz, dx);
load_current_density(J);

num::VectorField3D B = num::MagneticSolver::solve_magnetic_field(J);
```

This computes vector potential components from Poisson solves and returns
\f$\mathbf{B}=\nabla\times\mathbf{A}\f$.
