# PDE Module {#page_pde}

**Header:** `#include "pde/pde.hpp"`
**Namespace:** `num`, `num::pde`

Finite-difference PDE infrastructure: stencil operators, 3D field types, Crank-Nicolson ADI time-stepping, and explicit diffusion steps.

---

## Contents

| Header | What it provides |
|--------|-----------------|
| `pde/stencil.hpp` | 2D/3D Laplacians, gradient, divergence, curl, fiber sweeps |
| `pde/fields.hpp` | `ScalarField3D`, `VectorField3D`, `FieldSolver`, `MagneticSolver` |
| `pde/adi.hpp` | `CrankNicolsonADI` -- Strang-split CN solver for parabolic equations |
| `pde/diffusion.hpp` | `diffusion_step_2d`, `diffusion_step_2d_dirichlet` -- explicit Euler |

---

## Stencil operators (`pde/stencil.hpp`)

### 2D operators

```cpp
// Dirichlet BCs (zero on boundary), NxN grid row-major (idx = i*N + j)
void num::laplacian_stencil_2d(const Vector& x, Vector& y, int N);

// Periodic BCs
void num::laplacian_stencil_2d_periodic(const Vector& x, Vector& y, int N);
```

All three stencils return `h² ∇²u` — divide by h² to recover the Laplacian:

| Function | Points | Order |
|----------|--------|-------|
| `laplacian_stencil_2d` | 5 (cross) | O(h²), Dirichlet |
| `laplacian_stencil_2d_periodic` | 5 (cross) | O(h²), periodic |
| `laplacian_stencil_2d_4th` | 13 (extended cross) | O(h⁴), Dirichlet |

The standard 5-point stencil:

\f[y_{i,j} = x_{i+1,j} + x_{i-1,j} + x_{i,j+1} + x_{i,j-1} - 4\,x_{i,j}\f]

The 4th-order 13-point stencil:

\f[
  y_{i,j} = \frac{1}{12}\bigl(
    -x_{i-2,j} + 16\,x_{i-1,j} - 30\,x_{i,j} + 16\,x_{i+1,j} - x_{i+2,j}
    -x_{i,j-2} + 16\,x_{i,j-1}               + 16\,x_{i,j+1} - x_{i,j+2}
  \bigr)
\f]

### 2D grid utilities

```cpp
// Initialise an NxN field from a callable f(x,y) -> real
template<typename F>
void num::fill_grid(Vector& u, int N, double h, F&& f);

// Extract a row/column as a plottable Series (node k at (k+1)*h)
num::Series num::row_slice(const Vector& u, int N, double h, int row);
num::Series num::col_slice(const Vector& u, int N, double h, int col);
```

with Dirichlet or periodic treatment at the boundaries.

### 2D fiber sweeps

Used internally by `CrankNicolsonADI` and available directly:

```cpp
// For each column j: extract fiber psi[*,j], call f(fiber), write back
template<typename F>
void num::col_fiber_sweep(CVector& psi, int N, F&& f);

// For each row i: extract fiber psi[i,*], call f(fiber), write back
template<typename F>
void num::row_fiber_sweep(CVector& psi, int N, F&& f);
```

### 3D operators

All operate on `Grid3D` objects with central differences and one-sided stencils at boundaries:

```cpp
void num::neg_laplacian_3d(const Vector& v, Vector& Av, int nx, int ny, int nz, double inv_dx2);
void num::gradient_3d(const Grid3D& phi, Grid3D& gx, Grid3D& gy, Grid3D& gz);
void num::divergence_3d(const Grid3D& fx, const Grid3D& fy, const Grid3D& fz, Grid3D& div);
void num::curl_3d(const Grid3D& Ax, const Grid3D& Ay, const Grid3D& Az,
                  Grid3D& Bx, Grid3D& By, Grid3D& Bz);
```

`neg_laplacian_3d` computes \f$(-\nabla^2 x)\f$ with Dirichlet BCs (boundary rows = identity), giving an SPD system suitable for CG.

---

## 3D fields (`pde/fields.hpp`)

See [fields.md](fields.md) for full documentation. The classes live in `num::`:

- `ScalarField3D` -- uniform-grid scalar potential with trilinear sampling
- `VectorField3D` -- three `ScalarField3D` components
- `FieldSolver` -- static methods: `solve_poisson`, `gradient`, `divergence`, `curl`
- `MagneticSolver` -- static methods: `current_density`, `solve_magnetic_field`

---

## Crank-Nicolson ADI (`pde/adi.hpp`)

Prefactored Crank-Nicolson solver for 2D parabolic equations via Strang splitting.
Suitable for TDSE, heat equations with complex coefficients.

```cpp
num::CrankNicolsonADI adi(N, dt, h);
```

Factorises two `ComplexTriDiag` systems on construction (\f$O(N)\f$):
- `td_half_` -- for half-step sweeps (\f$\tau = dt/2\f$)
- `td_full_` -- for full-step sweep (\f$\tau = dt\f$)

### Sweep

```cpp
adi.sweep(psi, x_axis, tau);
```

Applies one CN sub-step along the chosen axis:

- `x_axis = true` -- processes each column fiber (x-direction)
- `x_axis = false` -- processes each row fiber (y-direction)
- Selects `td_half_` when `tau < 0.75*dt`, `td_full_` otherwise

### Strang splitting (TDSE)

```cpp
// Full time step:
adi.sweep(psi, true,  dt * 0.5);  // x, half-step
adi.sweep(psi, false, dt);         // y, full-step
adi.sweep(psi, true,  dt * 0.5);  // x, half-step
```

Each sweep solves the 1D tridiagonal system per fiber:

\f[\left(I - i\alpha\,\nabla^2_{1D}\right)\psi^{n+1} = \left(I + i\alpha\,\nabla^2_{1D}\right)\psi^n, \qquad \alpha = \frac{\tau}{4h^2}\f]

The LHS tridiagonal has sub/super-diagonal \f$-i\alpha\f$ and main diagonal \f$1 + 2i\alpha\f$;
it is factored once on construction and reused for every fiber.

---

## Explicit diffusion (`pde/diffusion.hpp`)

Forward Euler diffusion steps for 2D uniform grids. Von Neumann stability requires `coeff <= 0.25`.

```cpp
// Periodic BCs:  u += coeff * Lap_periodic(u)
num::pde::diffusion_step_2d(u, N, coeff);
num::pde::diffusion_step_2d(u, N, coeff, num::best_backend);

// Dirichlet BCs: u += coeff * Lap_dirichlet(u)
num::pde::diffusion_step_2d_dirichlet(u, N, coeff);
```

`coeff = alpha*dt/h^2` where \f$\alpha\f$ is the diffusion coefficient (e.g. kinematic viscosity \f$\nu\f$).

The explicit update is

\f[u^{n+1}_{i,j} = u^n_{i,j} + \text{coeff}\,\bigl(u^n_{i+1,j} + u^n_{i-1,j} + u^n_{i,j+1} + u^n_{i,j-1} - 4u^n_{i,j}\bigr)\f]

### Typical usage (Navier-Stokes viscosity)

```cpp
const double coeff = dt * nu / (h * h);
num::pde::diffusion_step_2d(u_star, N, coeff, num::best_backend);
num::pde::diffusion_step_2d(v_star, N, coeff, num::best_backend);
```

---

## Backward compatibility

`spatial/stencil.hpp` and `spatial/fields.hpp` are forwarding shims that include the new `pde/` headers. Existing code using the old include paths continues to work unchanged.
