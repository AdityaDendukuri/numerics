# Stencil Higher-Order Functions {#page_stencil_hof}

`include/pde/stencil.hpp` provides templated, higher-order utilities for the
most common grid operations in numerical PDE codes: finite-difference Laplacians,
operator-splitting fiber sweeps, and vector-calculus operators on 3D grids.
They eliminate the nested `for` loops that otherwise appear verbatim in every
physics application. The old path `include/spatial/stencil.hpp` is a forwarding
shim and continues to work.

---

## Motivation

Every PDE app in `apps/` originally contained variations of the same two patterns:

**Pattern 1 -- 5-point Laplacian stencil (2D, Dirichlet)**
```cpp
for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) {
        double lap = -4.0 * x[i*N+j];
        if (i > 0)   lap += x[(i-1)*N+j];
        if (i < N-1) lap += x[(i+1)*N+j];
        if (j > 0)   lap += x[i*N+j-1];
        if (j < N-1) lap += x[i*N+j+1];
        y[i*N+j] = -c * lap + V[i*N+j] * x[i*N+j];
    }
```
This appeared twice in TDSE (`compute_energy`, `ham_mv`) and in the NS pressure
Laplacian.

**Pattern 2 -- ADI fiber sweep (TDSE Crank-Nicolson)**
```cpp
for (int j = 0; j < N; ++j) {          // outer: iterate over fibers
    for (int i = 0; i < N; ++i)        // inner: build tridiagonal RHS
        rhs[i] = ... psi[i*N+j] ...;
    td.solve(rhs);
    for (int i = 0; i < N; ++i) psi[i*N+j] = rhs[i];
}
```
`sweep_x` and `sweep_y` in TDSE were identical modulo the axis index.

With `stencil.hpp` both patterns become single calls, and `sweep_x`/`sweep_y`
collapse into a shared `cn_sweep_` helper.

---

## 2D Functions

All 2D functions operate on a flat `BasicVector<T>` representing an \f$N \times N\f$
grid in **row-major order**: `data[i*N + j]` for row \f$i\f$, column \f$j\f$.
The template parameter `T` can be `num::real` (double) or `num::cplx`
(std::complex\<double\>).

### laplacian_stencil_2d -- Dirichlet boundaries

```cpp
template<typename T>
void num::laplacian_stencil_2d(const BasicVector<T>& x, BasicVector<T>& y, int N);
```

Computes the unnormalized 5-point Laplacian at each grid point:

\f[
y_{i,j} = x_{i+1,j} + x_{i-1,j} + x_{i,j+1} + x_{i,j-1} - 4\,x_{i,j}
\f]

Out-of-bounds neighbors contribute 0 (Dirichlet ghost cells).
To obtain the scaled operator \f$-\nabla^2_h x\f$ used in the Hamiltonian:

```cpp
num::CVector lap(N*N);
num::laplacian_stencil_2d(psi, lap, N);
const double c = 1.0 / (2.0 * h * h);
for (int k = 0; k < N*N; ++k)
    y[k] = -c * lap[k] + V[k] * psi[k];   // H = -(1/2)Lap + V
```

**Used by:** TDSE `compute_energy`, TDSE `ham_mv` (Lanczos Hamiltonian).

---

### laplacian_stencil_2d_periodic -- Periodic boundaries

```cpp
template<typename T>
void num::laplacian_stencil_2d_periodic(const BasicVector<T>& x, BasicVector<T>& y, int N);
```

Same formula as above, but out-of-bounds neighbors wrap modulo \f$N\f$.
The \f$j\f$-boundary rows (\f$j=0\f$ and \f$j=N-1\f$) are **peeled** out of
the inner loop so the interior range \f$j = 1 \ldots N-2\f$ contains no
modulo arithmetic and the compiler can auto-vectorize it (NEON/AVX-256).

Applying viscosity in one call (replaces the 15-line copy+double-loop in NS):

```cpp
num::Vector lap_u(N*N), lap_v(N*N);
num::laplacian_stencil_2d_periodic(u_star, lap_u, N);
num::laplacian_stencil_2d_periodic(v_star, lap_v, N);
num::axpy(c, lap_u, u_star, num::best_backend);   // u* += c * Lapu*
num::axpy(c, lap_v, v_star, num::best_backend);
```

For the pressure Poisson solve (negative Laplacian matvec in CG):

```cpp
auto neg_lap = [&](const num::Vector& p, num::Vector& out) {
    num::laplacian_stencil_2d_periodic(p, out, N);
    num::scale(out, -inv_h2);
};
```

**Used by:** NS `apply_diffusion`, NS `solve_pressure`.

---

### col_fiber_sweep / row_fiber_sweep -- ADI/CN sweeps

```cpp
template<typename T, typename F>
void num::col_fiber_sweep(BasicVector<T>& data, int N, F&& f);

template<typename T, typename F>
void num::row_fiber_sweep(BasicVector<T>& data, int N, F&& f);
```

For each fiber (column or row), these extract the \f$N\f$ values into a
`std::vector<T>`, invoke `f(fiber)`, then write back.  The callable `f` has
signature `void(std::vector<T>&)` and modifies the fiber in-place.

`col_fiber_sweep` iterates over columns \f$j = 0 \ldots N-1\f$ and extracts
the x-direction fiber `data[0..N-1, j]` -- used for Crank-Nicolson sweeps
along \f$x\f$ in TDSE.

`row_fiber_sweep` iterates over rows \f$i = 0 \ldots N-1\f$ and extracts
`data[i, 0..N-1]` -- used for sweeps along \f$y\f$.

**Collapsing TDSE sweep_x and sweep_y into one helper:**

```cpp
void TDSESolver::cn_sweep_(bool x_axis, double tau) {
    const num::ComplexTriDiag& td = (tau < dt * 0.75) ? td_half_ : td_full_;
    const cplx ia(0.0, tau / (4.0 * h * h));
    const cplx diag(1.0, -2.0 * tau / (4.0 * h * h));

    auto apply = [&](std::vector<cplx>& fiber) {
        std::vector<cplx> rhs(N);
        for (int i = 0; i < N; ++i) {
            cplx prev = (i > 0)   ? fiber[i-1] : cplx{};
            cplx next = (i < N-1) ? fiber[i+1] : cplx{};
            rhs[i] = ia * prev + diag * fiber[i] + ia * next;
        }
        td.solve(rhs);
        fiber = std::move(rhs);
    };

    if (x_axis) num::col_fiber_sweep(psi, N, apply);
    else        num::row_fiber_sweep(psi, N, apply);
}

void TDSESolver::sweep_x(double tau) { cn_sweep_(true,  tau); }
void TDSESolver::sweep_y(double tau) { cn_sweep_(false, tau); }
```

**Used by:** TDSE `sweep_x`, `sweep_y`.

---

## 3D Functions

All 3D functions operate on `num::Grid3D` objects (uniform Cartesian grid,
flat layout `idx = k*ny*nx + j*nx + i`, stored in `Grid3D::data_`).
Central differences are used throughout; **one-sided differences** are applied
at domain boundaries.

### neg_laplacian_3d -- 7-point stencil

```cpp
void num::neg_laplacian_3d(const Vector& x, Vector& y,
                            int nx, int ny, int nz, double inv_dx2);
```

Computes \f$y = -\Delta_h x\f$ on a flat vector in Grid3D layout:

\f[
y_{i,j,k} = \frac{6\,x_{i,j,k} - \sum_{\pm} \bigl(x_{i\pm1,j,k} + x_{i,j\pm1,k} + x_{i,j,k\pm1}\bigr)}{h^2}
\f]

Boundary nodes satisfy \f$y = x\f$ (identity row), making the operator SPD
when paired with a zero-Dirichlet RHS -- suitable for direct use with `cg_matfree`.

Replacing the 16-line triple-loop matvec in EM `solve_poisson`:

```cpp
auto matvec = [&](const num::Vector& v, num::Vector& Av) {
    num::neg_laplacian_3d(v, Av, nx, ny, nz, inv_dx2);
};
auto result = num::cg_matfree(matvec, b, xv, tol, max_iter);
```

**Used by:** EM `FieldSolver::solve_poisson`.

---

### gradient_3d

```cpp
void num::gradient_3d(const Grid3D& phi,
                       Grid3D& gx, Grid3D& gy, Grid3D& gz);
```

Computes \f$(\nabla\phi)_{i,j,k}\f$ via second-order central differences,
one-sided at boundaries:

\f[
(\nabla\phi)^x_{i,j,k} = \frac{\phi_{i+1,j,k} - \phi_{i-1,j,k}}{2h}, \quad
(\nabla\phi)^y_{i,j,k} = \frac{\phi_{i,j+1,k} - \phi_{i,j-1,k}}{2h}, \quad
(\nabla\phi)^z_{i,j,k} = \frac{\phi_{i,j,k+1} - \phi_{i,j,k-1}}{2h}
\f]

```cpp
VectorField3D FieldSolver::gradient(const ScalarField3D& phi) {
    VectorField3D out(phi.nx(), phi.ny(), phi.nz(), phi.dx(),
                      phi.ox(), phi.oy(), phi.oz());
    num::gradient_3d(phi.grid(), out.x.grid(), out.y.grid(), out.z.grid());
    return out;
}
```

**Used by:** EM `FieldSolver::gradient`, `MagneticSolver::current_density`.

---

### divergence_3d

```cpp
void num::divergence_3d(const Grid3D& fx, const Grid3D& fy, const Grid3D& fz,
                         Grid3D& out);
```

Computes \f$\nabla \cdot \mathbf{f}\f$ via central differences:

\f[
(\nabla \cdot \mathbf{f})_{i,j,k} =
  \frac{f^x_{i+1,j,k} - f^x_{i-1,j,k}}{2h}
+ \frac{f^y_{i,j+1,k} - f^y_{i,j-1,k}}{2h}
+ \frac{f^z_{i,j,k+1} - f^z_{i,j,k-1}}{2h}
\f]

**Used by:** EM `FieldSolver::divergence`.

---

### curl_3d

```cpp
void num::curl_3d(const Grid3D& ax, const Grid3D& ay, const Grid3D& az,
                   Grid3D& bx, Grid3D& by, Grid3D& bz);
```

Computes \f$\mathbf{B} = \nabla \times \mathbf{A}\f$ component-wise:

\f[
B^x = \partial_y A^z - \partial_z A^y, \quad
B^y = \partial_z A^x - \partial_x A^z, \quad
B^z = \partial_x A^y - \partial_y A^x
\f]

All partial derivatives are second-order central differences, one-sided at
boundaries.

**Used by:** EM `FieldSolver::curl` (and implicitly `MagneticSolver::solve_magnetic_field`
via `B = gradxA`).

---

## Where Each Function Is Used

| Function | 2D/3D | BC | App | Purpose |
|---|---|---|---|---|
| `laplacian_stencil_2d` | 2D | Dirichlet | TDSE | Hamiltonian matvec, energy expectation |
| `laplacian_stencil_2d_periodic` | 2D | Periodic | NS | Viscosity step, pressure Poisson |
| `col_fiber_sweep` | 2D | -- | TDSE | Crank-Nicolson sweep along x |
| `row_fiber_sweep` | 2D | -- | TDSE | Crank-Nicolson sweep along y |
| `neg_laplacian_3d` | 3D | Dirichlet | EM | Poisson solve matvec |
| `gradient_3d` | 3D | one-sided | EM | Electric field, current density |
| `divergence_3d` | 3D | one-sided | EM | Divergence diagnostic |
| `curl_3d` | 3D | one-sided | EM | Magnetic field from vector potential |

---

## Adding a New App

Any new 2D app operating on an \f$N \times N\f$ interior grid with Dirichlet or
periodic BCs can directly call these functions without reimplementing stencil
loops.  For a heat equation solver:

```cpp
// Explicit Euler: u^{n+1} = u^n + (alpha * dt / h^2) * Lap_periodic(u^n)
void heat_step(num::Vector& u, int N, double alpha, double dt, double h) {
    num::Vector lap(N * N);
    num::laplacian_stencil_2d_periodic(u, lap, N);
    num::axpy(alpha * dt / (h * h), lap, u, num::best_backend);
}
```

For a 3D Poisson solve with Dirichlet BCs:

```cpp
// -Lapphi = rho  (phi = 0 on boundary)
num::SolverResult solve_poisson_3d(num::Grid3D& phi,
                                    const num::Grid3D& rho,
                                    double tol = 1e-8) {
    int nx = phi.nx(), ny = phi.ny(), nz = phi.nz();
    double inv_dx2 = 1.0 / (phi.dx() * phi.dx());
    int N = nx * ny * nz;

    num::Vector b = rho.to_vector();   // RHS
    num::Vector x = phi.to_vector();   // initial guess

    auto matvec = [&](const num::Vector& v, num::Vector& Av) {
        num::neg_laplacian_3d(v, Av, nx, ny, nz, inv_dx2);
    };
    auto res = num::cg_matfree(matvec, b, x, tol, 1000);
    phi.from_vector(x);
    return res;
}
```
