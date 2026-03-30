# 3D Field Types and Solvers {#page_fields}

`include/pde/fields.hpp` and `src/pde/fields.cpp` provide general-purpose
3D scalar and vector field types together with PDE field solvers for Poisson,
gradient, divergence, and curl. The old path `include/spatial/fields.hpp` is a
forwarding shim and continues to work.

These classes previously lived in `apps/em_demo/field.hpp/cpp` as `physics::` types.
Moving them into `num::` makes them available to any 3D app without copy-pasting.

---

## Types

### ScalarField3D

```cpp
class num::ScalarField3D {
public:
    ScalarField3D(int nx, int ny, int nz, float dx,
                  float ox=0, float oy=0, float oz=0);

    num::Grid3D&       grid();
    const num::Grid3D& grid() const;

    int   nx(), ny(), nz();
    float dx(), ox(), oy(), oz();

    void  set(int i, int j, int k, double v);
    void  fill(double v);

    /// Trilinear interpolation at world position (x,y,z).
    /// Returns 0 outside the grid domain.
    float sample(float x, float y, float z) const;
};
```

Wraps a `num::Grid3D` with a world-space origin `(ox, oy, oz)`.
`sample()` converts world coordinates to grid coordinates and performs
trilinear interpolation across the 8 surrounding nodes.

---

### VectorField3D

```cpp
struct num::VectorField3D {
    ScalarField3D x, y, z;   // all on the same grid

    VectorField3D(int nx, int ny, int nz, float dx,
                  float ox=0, float oy=0, float oz=0);

    std::array<float,3> sample(float px, float py, float pz) const;
    void scale(float s);
};
```

Three co-located components.  `sample()` returns
`{x.sample(p), y.sample(p), z.sample(p)}` via trilinear interpolation.

---

## FieldSolver

Static PDE utilities.  All methods operate on `ScalarField3D` / `VectorField3D`
in-place or return new field objects.

```cpp
class num::FieldSolver {
public:
    /// Solve Lapphi = source  (phi=0 Dirichlet on all boundaries).
    static SolverResult solve_poisson(ScalarField3D& phi,
                                      const ScalarField3D& source,
                                      double tol = 1e-6, int max_iter = 500);

    /// gradphi via central differences (one-sided at boundaries).
    static VectorField3D gradient(const ScalarField3D& phi);

    /// grad*f via central differences.
    static ScalarField3D divergence(const VectorField3D& f);

    /// gradxA via central differences (one-sided at boundaries).
    static VectorField3D curl(const VectorField3D& A);
};
```

`solve_poisson` internally calls `num::neg_laplacian_3d` (see
\ref page_stencil_hof) and `num::cg_matfree`.  The SPD operator (-Lap) with identity
rows on the boundary means the Dirichlet system is SPD -> CG converges.

---

## MagneticSolver

```cpp
class num::MagneticSolver {
public:
    static constexpr double MU0 = 1.2566370614e-6;  // mu_0 [H/m]

    /// J = -sigmagradphi [A/m^2]
    static VectorField3D current_density(const ScalarField3D& sigma,
                                          const ScalarField3D& phi);

    /// Solve for B given J.
    /// Solves LapA = -mu_0J (Coulomb gauge, Dirichlet) then returns B = gradxA.
    static VectorField3D solve_magnetic_field(const VectorField3D& J,
                                               double tol = 1e-6, int max_iter = 500);
};
```

`solve_magnetic_field` runs three independent `solve_poisson` calls (one per
component of **A**) then applies `curl`.

---

## Typical Workflow

```cpp
// Electric field from charge distribution
num::ScalarField3D phi(32,32,32, 0.05f);
num::ScalarField3D rho(32,32,32, 0.05f);
rho.set(16,16,16, charge_density);
num::FieldSolver::solve_poisson(phi, rho);
num::VectorField3D E = num::FieldSolver::gradient(phi);
E.scale(-1.0f);   // E = -gradphi

// Per-particle force
for (auto& p : particles) {
    auto [ex, ey, ez] = E.sample(p.x, p.y, p.z);
    p.ax += charge * ex / p.mass;
}
```

```cpp
// Magnetic field from DC current
num::ScalarField3D sigma(nx, ny, nz, dx);
num::ScalarField3D Vphi(nx, ny, nz, dx);
// ... fill sigma, solve ElectricSolver::solve_potential(Vphi, sigma, bcs) ...
num::VectorField3D J = num::MagneticSolver::current_density(sigma, Vphi);
num::VectorField3D B = num::MagneticSolver::solve_magnetic_field(J);
```

---

## Where Each Type Is Used

| Type / Method | App | Purpose |
|---|---|---|
| `ScalarField3D`, `VectorField3D` | EM demo | Potential phi, conductivity sigma, E and B fields |
| `FieldSolver::solve_poisson` | EM demo | Lapphi = rho for electrostatic potential |
| `FieldSolver::gradient` | EM demo | E = -gradphi, J component from gradient |
| `FieldSolver::curl` | EM demo | B = gradxA |
| `MagneticSolver::current_density` | EM demo | J = -sigmagradphi |
| `MagneticSolver::solve_magnetic_field` | EM demo | Vector Poisson + curl for B |

The EM-specific `ElectricSolver` (variable-conductivity div(sigmagradphi)=0 with
electrode BCs and Joule heating) remains in `apps/em_demo/field.hpp`.
