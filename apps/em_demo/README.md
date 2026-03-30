<!--! @page page_app_em Electromagnetic Field Demo -->

# Electromagnetic Field Demo

Batch renderer for steady-state current flow through an aluminium rod (DC electrostatics) coupled with magnetostatics from the induced current. All Poisson problems are solved with `num::cg_matfree` on a $32^3$ voxel grid.

Preset scene: electrostatic field solved once for a fixed conductor geometry. Camera orbits 360 degrees over 400 frames. Exports 400 frames at 30 fps.

---

## Physical Model

The simulation couples two classical field theories on a uniform 3D grid of $N_x \times N_y \times N_z = 32^3$ cells with spacing $\Delta x = 0.05$ m (domain approximately $1.6^3$ m^3).

### Electrostatics -- Steady Current Flow

With no time variation and no volume charge sources, the electric potential $\varphi$ satisfies

$$\nabla \cdot (\sigma \nabla \varphi) = 0$$

where $\sigma(\mathbf{x})$ is the spatially varying conductivity. Boundary conditions:

- **Dirichlet** (electrodes): $\varphi = +1$ V at the rod's top face; $\varphi = 0$ V at the bottom.
- **Neumann** (all other surfaces): $\partial\varphi/\partial n = 0$ (no current escapes).

The current density follows Ohm's law:

$$\mathbf{J} = -\sigma \nabla\varphi$$

and the Joule heating power density is $Q = \sigma|\nabla\varphi|^2$.

### Magnetostatics -- Vector Potential

In the Coulomb gauge, the magnetic vector potential $\mathbf{A}$ satisfies three independent scalar Poisson problems:

$$\nabla^2 \mathbf{A} = -\mu_0 \mathbf{J}, \qquad \mu_0 = 4\pi \times 10^{-7}\;\text{H/m}$$

The magnetic flux density is then $\mathbf{B} = \nabla \times \mathbf{A}$, computed by central differences on the grid.

---

## Geometry

| Component | Description |
|-----------|-------------|
| Rod | Vertical cylinder, radius 4 cells (0.2 m), full domain height |
| Rod conductivity | $\sigma = 1.0$ S/m |
| Background | $\sigma = 10^{-6}$ S/m (near-insulator) |
| Top electrode | $\varphi = +1.0$ V, rod cross-section at $y = N_y - 1$ |
| Bottom electrode | $\varphi = 0.0$ V, rod cross-section at $y = 0$ |

---

## Numerical Method

All field equations share the same structure: a **symmetric positive definite sparse linear system** solved by `num::cg_matfree`.

### Variable-Coefficient Potential Solve

The finite-difference discretisation of $\nabla \cdot (\sigma \nabla \varphi) = 0$ at interior node $(i,j,k)$ is:

$$\sum_{\text{face}} \sigma_{\text{face}}\,\frac{\varphi_{\text{neighbour}} - \varphi_{i,j,k}}{\Delta x^2} = 0$$

where $\sigma_{\text{face}}$ is the harmonic mean of the two adjacent cell conductivities. Dirichlet nodes are enforced by a **penalty method**: the diagonal entry for electrode node $n$ is set to $10^{10}$ and the right-hand side to $10^{10}\varphi_{\text{prescribed}}$.

### Magnetic Poisson Solves

The three scalar problems $\nabla^2 A_\alpha = -\mu_0 J_\alpha$ ($\alpha \in \{x,y,z\}$) are solved sequentially with the same matrix-free CG. The 6-point Laplacian stencil is:

$$(\mathbf{A}\mathbf{p})_{ijk} = \frac{6p_{ijk} - p_{i\pm1} - p_{j\pm1} - p_{k\pm1}}{\Delta x^2}$$

with Dirichlet $\mathbf{A} = 0$ on domain boundaries.

---

## Numerics Library Integration

| Feature | Where used |
|---------|-----------|
| `num::ScalarField3D` (`pde/fields.hpp`) | Grid-backed scalar fields for $\varphi$, $\sigma$, $A_x$/$A_y$/$A_z$ |
| `num::VectorField3D` (`pde/fields.hpp`) | Three-component field for $\mathbf{J}$ and $\mathbf{B}$ |
| `num::FieldSolver::solve_poisson` | Matrix-free CG solve of $\Delta\varphi = f$ with Dirichlet BCs |
| `num::MagneticSolver::current_density` | Computes $\mathbf{J} = -\sigma\nabla\varphi$ from field data |
| `num::MagneticSolver::solve_magnetic_field` | Solves $\Delta\mathbf{A} = -\mu_0\mathbf{J}$; returns $\mathbf{B} = \nabla\times\mathbf{A}$ |
| `num::cg_matfree` | Underlying CG solver called by `FieldSolver` and `ElectricSolver` |
| `num::Grid3D` | Underlying 3D array backing each `ScalarField3D` |
| `num::Vector` | Flattened field arrays passed to the CG solver |

The solver assembles no explicit matrix. The matvec is a lambda capturing the conductivity field, evaluated in $O(N^3)$ per call. CG converges in $O(\sqrt{\kappa})$ iterations where $\kappa$ is the condition number -- typically fewer than 100 iterations for the $32^3$ grid.

---

## Visualisation

- **Domain:** White wireframe cube.
- **Aluminium rod:** Semi-transparent grey cylinder; red cap (anode) / blue cap (cathode).
- **$\mathbf{B}$ field arrows:** Quiver plot on a $32^3$ grid subsampled every `STRIDE = 4` cells.
  - Arrow direction: $\hat{\mathbf{B}}$.
  - Arrow length: $0.15$ to $1.0 \times 2.2\Delta x$ on a log scale of $|\mathbf{B}|$.
  - Arrow colour: HSV hue from 0 degrees (blue, low $|\mathbf{B}|$) to 240 degrees (red, high $|\mathbf{B}|$).
  - Arrows below 0.2% of $B_{\max}$ are hidden to reduce clutter.

---

## Project layout

    include/em_demo/sim.hpp   -- ElectricSolver, ElectrodeBC declarations
    src/sim.cpp               -- CG Poisson solve, field computation
    main.cpp                  -- batch frame exporter (fixed scene, 400 frames)
    make_video.sh             -- ffmpeg compositor -> em_demo.mp4

---

## Running

```bash
cmake -B build -DNUMERICS_BUILD_EM_DEMO=ON
cmake --build build --target em_demo
./build/apps/em_demo/em_demo
bash apps/em_demo/make_video.sh
```

> **macOS note:** The `em_demo` target defines a `DOMAIN` macro that conflicts with the system `<math.h>` constant on macOS. Rename the macro to `SIM_DOMAIN` before building if you encounter this.

---

## References

- J. D. Jackson, *Classical Electrodynamics*, 3rd ed., Wiley (1999) -- magnetostatics, dipole field
- O. C. Zienkiewicz & R. L. Taylor, *The Finite Element Method*, 6th ed., Butterworth-Heinemann (2005) -- penalty Dirichlet BCs
- Y. Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed., SIAM (2003) -- matrix-free CG
