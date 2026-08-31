# Partial Differential Equations {#page_pde}

Finite difference stencils, sparse Laplacian assembly, matrix-free backward Euler operators, and fast direct Poisson solvers.

---

## 1. 2D Finite Difference Stencils

### Explicit Diffusion Stepping (\f$\frac{\partial u}{\partial t} = D \nabla^2 u\f$)

```cpp
#include <numerics.hpp>

constexpr int n = 128;
constexpr double h = 1.0 / (n + 1);
num::Grid2D grid{n, h};

num::ScalarField2D u(grid, [](double x, double y) {
    return std::exp(-80.0 * ((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5)));
});

double coeff = diffusivity * dt / (h * h);

// 2nd-order 5-point stencil (periodic boundaries)
num::pde::diffusion_step_2d(u.vec(), n, coeff);

// 2nd-order 5-point stencil (homogeneous Dirichlet: u = 0 on boundary)
num::pde::diffusion_step_2d_dirichlet(u, coeff);

// 4th-order centered stencil (homogeneous Dirichlet)
num::pde::diffusion_step_2d_4th_dirichlet(u, coeff, num::backend::dflt);
```

---

## 2. Discrete Laplacian and Implicit Stepping

### 2D Poisson Laplacian (num::pde::laplacian_sparse_2d)
Assembles negative-definite 2D discrete Laplacian matrix \f$L \in \mathbb{R}^{n^2 \times n^2}\f$ in CSR format with 5-point stencil.

```cpp
num::SparseMatrix L = num::pde::laplacian_sparse_2d(n);
```

### Backward Euler Matrix (num::pde::backward_euler_matrix)
Assembles the sparse matrix \f$I - c L\f$ where \f$c = \frac{D \Delta t}{h^2}\f$.

```cpp
num::SparseMatrix system = num::pde::backward_euler_matrix(grid, coeff);
```

### Backward Euler Operator (num::pde::backward_euler_operator)
Constructs a matrix-free linear operator for \f$y = (I - c L) x\f$ with compile-time `SPDOperator` tag and zero internal allocations.

```cpp
auto op = num::pde::backward_euler_operator(grid, coeff);
static_assert(num::SPDOperator<decltype(op)>);

num::Vector rhs(grid.size(), 1.0);
num::Vector sol(grid.size(), 0.0);
num::cg(op, rhs, sol, 1e-8); // Direct solve without matrix assembly
```

---

## 3. Fast Direct Poisson Solvers (DST-I)

Solves \f$-\nabla^2 \phi = \rho\f$ with \f$\phi|_{\partial \Omega} = 0\f$ on \f$[0, 1]^2\f$ in \f$\mathcal{O}(n^2 \log n)\f$ time using Discrete Sine Transforms.

```cpp
num::Matrix source(n, n, 0.0);
// fill source density rho...

// 1. Exact 5-point finite difference eigenvalues
num::Matrix phi_fd = num::pde::poisson2d_fd(source, n);

// 2. Continuous Laplacian modal eigenvalues
num::Matrix phi_spectral = num::pde::poisson2d(source, n);
```

---

## 4. Crank–Nicolson ADI (num::CrankNicolsonADI)

Alternating Direction Implicit sweeps for 2D parabolic equations:

```cpp
num::CrankNicolsonADI adi(n, dt, h);
num::CVector psi(n * n);

adi.sweep(psi, /*column_sweep=*/true, dt);
adi.sweep(psi, /*column_sweep=*/false, dt);
```

---

## Complete Example

@example 06_pde_poisson_solver.cpp

