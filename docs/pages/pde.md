# PDE Examples {#page_pde}

## Grid and Initial Field

```cpp
#include <numerics.hpp>

constexpr int n = 128;
constexpr double h = 1.0 / (n + 1);
num::Grid2D grid{n, h};

num::ScalarField2D temperature(grid, [](double x, double y) {
    return std::exp(-80.0 * ((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5)));
});
```

## Periodic Diffusion Step

```cpp
double coefficient = diffusivity * dt / (h * h);
num::pde::diffusion_step_2d(temperature.vec(), n, coefficient); // Periodic stencil.
```

## Dirichlet Diffusion Step

```cpp
num::pde::diffusion_step_2d_dirichlet(temperature, coefficient); // Zero boundary values.
```

## Fourth-Order Diffusion Step

```cpp
num::pde::diffusion_step_2d_4th_dirichlet(
    temperature, coefficient, num::best_backend); // Fourth-order interior stencil.
```

## Sparse Laplacian

```cpp
num::SparseMatrix laplacian = num::pde::laplacian_sparse_2d(n); // Five-point operator.
```

The matrix acts on a row-major vector of `n * n` interior values.

## Backward Euler Matrix

```cpp
num::SparseMatrix system =
    num::pde::backward_euler_matrix(grid, coefficient); // I - coefficient * L.
```

## Backward Euler Operator

```cpp
auto system = num::pde::backward_euler_operator(grid, coefficient); // Owns its sparse matrix.
num::idx rows = system.rows();
const num::SparseMatrix& matrix = system.matrix();
```

The operator carries symmetric and positive-definite properties for CG dispatch.

## Reusable CG Solver

```cpp
num::SparseMatrix system = num::pde::backward_euler_matrix(grid, coefficient);
num::LinearSolver solve_step = num::pde::make_cg_solver(system, 1e-8); // Captures system by reference.

num::ode::advance(temperature, solve_step, {.nstep = 100, .dt = dt});
```

`system` must outlive `solve_step`.

## Custom Implicit Solver

```cpp
auto system = num::pde::backward_euler_operator(grid, coefficient);

num::LinearSolver solve_step = [&](const num::Vector& rhs, num::Vector& x) {
    return num::cg(system, rhs, x, 1e-8, 500);
};
```

## Applying a Laplacian Stencil

```cpp
num::Vector laplacian(temperature.size());
num::laplacian_stencil_2d(temperature.vec(), laplacian, n); // Zero exterior values.
```

```cpp
num::laplacian_stencil_2d_periodic(
    temperature.vec(), laplacian, n); // Wrap both grid axes.
```

```cpp
num::laplacian_stencil_2d_4th(
    temperature.vec(), laplacian, n); // Fourth-order centered stencil.
```

## Periodic Interpolation

```cpp
double value = num::sample_2d_periodic(
    temperature, -0.1, 0.3, 0.0, 0.0); // Coordinates wrap into the grid domain.
```

## Fiber Sweeps

```cpp
num::row_fiber_sweep(temperature.vec(), n, [](std::vector<double>& row) {
    filter(row); // Receives one copied row and writes it back.
});
```

```cpp
num::col_fiber_sweep(temperature.vec(), n, [](std::vector<double>& column) {
    filter(column); // Receives one copied column and writes it back.
});
```

## Crank-Nicolson ADI Sweeps

```cpp
num::CrankNicolsonADI adi(n, dt, h); // Pre-factor half-step and full-step systems.
num::CVector wavefunction(n * n);

adi.sweep(wavefunction, true, dt);  // Sweep columns.
adi.sweep(wavefunction, false, dt); // Sweep rows.
```

## Discrete Poisson Solve

```cpp
num::Matrix source(n, n, 0.0);
num::Matrix potential = num::pde::poisson2d_fd(source, n); // Finite-difference eigenvalues.
```

## Spectral Poisson Solve

```cpp
num::Matrix potential = num::pde::poisson2d(source, n); // Continuous sine-mode eigenvalues.
```

Both Poisson solvers use homogeneous Dirichlet boundaries on the unit square.
See @ref page_poisson for the full derivation.

## Complete Program

@example 06_pde_poisson_solver.cpp
