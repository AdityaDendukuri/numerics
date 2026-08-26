# Partial Differential Equations {#page_pde}

The `pde` module provides finite difference stencils, matrix-free grid operators, sparse Laplacian assembly, implicit parabolic steppers, and fast direct Poisson solvers.

---

## 1. 2D Diffusion & Parabolic Systems

Models transient heat conduction and mass diffusion:

\f[
\frac{\partial u}{\partial t} = D \nabla^2 u = D \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right)
\f]

```cpp
#include <numerics.hpp>

constexpr int n = 128;
constexpr double h = 1.0 / (n + 1);
num::Grid2D grid{n, h};

num::ScalarField2D temperature(grid, [](double x, double y) {
    return std::exp(-80.0 * ((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5)));
});
```

### Explicit Finite-Difference Stencils

#### 2nd-Order 5-Point Centered Stencil

\f[
(\nabla^2 u)_{i,j} \approx \frac{u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - 4 u_{i,j}}{h^2} + \mathcal{O}(h^2)
\f]

```cpp
double coefficient = diffusivity * dt / (h * h);
num::pde::diffusion_step_2d(temperature.vec(), n, coefficient);           // Periodic boundaries
num::pde::diffusion_step_2d_dirichlet(temperature, coefficient);          // Homogeneous Dirichlet boundaries (u = 0)
```

#### 4th-Order Centered Stencil

\f[
\left(\frac{\partial^2 u}{\partial x^2}\right)_i \approx \frac{-u_{i+2} + 16u_{i+1} - 30u_i + 16u_{i-1} - u_{i-2}}{12 h^2} + \mathcal{O}(h^4)
\f]

```cpp
num::pde::diffusion_step_2d_4th_dirichlet(temperature, coefficient, num::backend::dflt);
```

---

## 2. Sparse Discrete Laplacian & Backward Euler

### 5-Point Sparse Laplacian Matrix (\f$L\f$)

Assembles the negative-definite 2D discrete Laplacian \f$L \in \mathbb{R}^{n^2 \times n^2}\f$ with pentadiagonal Kronecker structure \f$L = I \otimes T + T \otimes I\f$:

\f[
L = \frac{1}{h^2} \begin{bmatrix}
T & I & & \\
I & T & \ddots & \\
& \ddots & \ddots & I \\
& & I & T
\end{bmatrix}, \qquad T = \begin{bmatrix}
-4 & 1 & & \\
1 & -4 & \ddots & \\
& \ddots & \ddots & 1 \\
& & 1 & -4
\end{bmatrix}
\f]

```cpp
num::SparseMatrix laplacian = num::pde::laplacian_sparse_2d(n); // CSR sparse matrix of size n^2 x n^2
```

### Implicit Backward Euler System (\f$I - c L\f$)

Unconditionally stable implicit stepping \f$(I - c L) u^{n+1} = u^n\f$ where \f$c = \frac{D \Delta t}{h^2}\f$:

```cpp
num::SparseMatrix system = num::pde::backward_euler_matrix(grid, coefficient); // I - c*L
```

### Matrix-Free Backward Euler Operator

Evaluates \f$y = (I - c L) x\f$ on the fly with \f$\mathcal{O}(1)\f$ memory overhead, carrying compile-time Symmetric Positive Definite (`SPDOperator`) tags:

```cpp
auto system = num::pde::backward_euler_operator(grid, coefficient); // Zero memory allocation
num::LinearSolver solve_step = num::pde::make_cg_solver(system.matrix(), 1e-8);

num::ode::advance(temperature, solve_step, {.nstep = 100, .dt = dt});
```

---

## 3. Fast Poisson Solvers (FFT / DST-I)

Solves the elliptic Poisson equation with homogeneous Dirichlet boundary conditions on \f$[0, 1]^2\f$:

\f[
-\nabla^2 \phi = \rho, \qquad \phi|_{\partial \Omega} = 0
\f]

Using Discrete Sine Transforms (DST-I), the exact modal decoupling solves in \f$\mathcal{O}(n^2 \log n)\f$:

\f[
\hat{\phi}_{m,n} = \frac{\hat{\rho}_{m,n}}{\lambda_{m,n}}, \qquad \lambda_{m,n} = \frac{4}{h^2} \left[ \sin^2\left(\frac{m \pi}{2(n+1)}\right) + \sin^2\left(\frac{n \pi}{2(n+1)}\right) \right]
\f]

```cpp
num::Matrix source(n, n, 0.0);
// Fill source density rho...

num::Matrix potential_fd = num::pde::poisson2d_fd(source, n); // Exact 5-point stencil eigenvalues
num::Matrix potential_spectral = num::pde::poisson2d(source, n); // Continuous Laplacian eigenvalues
```

---

## 4. Alternating Direction Implicit (ADI) Sweeps

Crank–Nicolson ADI splits the 2D operator into sequential 1D tridiagonal sweeps:

\f[
\left(I - \frac{\Delta t}{2} \mathcal{L}_x\right) u^{n+1/2} = \left(I + \frac{\Delta t}{2} \mathcal{L}_y\right) u^n, \qquad \left(I - \frac{\Delta t}{2} \mathcal{L}_y\right) u^{n+1} = \left(I + \frac{\Delta t}{2} \mathcal{L}_x\right) u^{n+1/2}
\f]

```cpp
num::CrankNicolsonADI adi(n, dt, h);
num::CVector wavefunction(n * n);

adi.sweep(wavefunction, /*column_sweep=*/true, dt);
adi.sweep(wavefunction, /*column_sweep=*/false, dt);
```

---

## Complete Example

@example 06_pde_poisson_solver.cpp
