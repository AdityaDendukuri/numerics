# PDE Examples {#page_pde}

Finite-difference routines live under `num::pde` and are included by
`numerics.hpp`.

## Explicit Diffusion Step

```cpp
constexpr int N = 128;
constexpr double h = 1.0 / (N + 1);
constexpr double dt = 0.25 * h * h;
constexpr double kappa = 1.0;

num::Vector u(N * N, 0.0);
initialise_gaussian(u, N, h);

num::pde::diffusion_step_2d_dirichlet(
    u, N, kappa * dt / (h * h), num::best_backend);
```

## Backward Euler Heat Solve

```cpp
num::Grid2D grid{N, h};
double coeff = kappa * dt / (h * h);

num::SparseMatrix A = num::pde::backward_euler_matrix(grid, coeff);
num::operators::SparseOp Aop(A);

num::LinearSolver solver = [&](const num::Vector& rhs, num::Vector& x) {
    return num::cg(num::operators::assume_spd(Aop), rhs, x, 1e-8, 500);
};

num::ScalarField2D u(grid, initial_condition);
num::ode::advance(u, solver, {.nstep = 100, .dt = dt});
```

The same system can be built as an owning SPD operator:

```cpp
auto A = num::pde::backward_euler_operator(grid, coeff);

num::LinearSolver solver = [&](const num::Vector& rhs, num::Vector& x) {
    return num::cg(A, rhs, x, 1e-8, 500);
};
```

The linear system for one step is

\f[
    (I - \Delta t\,\kappa L_h)u^{n+1}=u^n .
\f]

## ADI Diffusion

```cpp
num::CrankNicolsonADI stepper(N, h, kappa, dt);

for (int step = 0; step < 200; ++step) {
    stepper.step(u.vec());
}
```

## Poisson Solve

For a square Dirichlet problem on an \f$N\times N\f$ interior grid:

```cpp
num::Matrix f(N, N, 0.0);
fill_rhs(f, N);

num::Matrix u = num::pde::poisson2d(f, N);
```

See @subpage page_poisson for the DST-based Poisson example.
