<!--! @page page_app_fluid 2D SPH Fluid Simulation -->

# 2D SPH Fluid Simulation

Batch renderer for Weakly Compressible Smoothed Particle Hydrodynamics (WCSPH) in 2D with coupled heat transport. The solver dispatches to sequential (Newton's-third-law pairs) or OpenMP (per-particle) backends from the numerics library.

Preset scene: dam break -- left-half water column collapses under gravity. Particles coloured by temperature (blue=cold, red=hot). Exports 600 frames at 30 fps.

---

## Physical Model

Each fluid particle $i$ carries position $\mathbf{x}_i$, velocity $\mathbf{v}_i$, density $\rho_i$, pressure $p_i$, and temperature $T_i$. The governing equations are the weakly compressible Navier-Stokes equations discretised by SPH:

$$\frac{D\rho_i}{Dt} = \sum_j m_j \mathbf{v}_{ij} \cdot \nabla_i W_{ij}$$

$$\frac{D\mathbf{v}_i}{Dt} = -\sum_j m_j \left(\frac{p_i}{\rho_i^2} + \frac{p_j}{\rho_j^2}\right)\nabla_i W_{ij} + \mu \sum_j m_j \frac{\mathbf{v}_{ij}}{\rho_i \rho_j} \nabla^2 W_{ij} + \mathbf{g}$$

$$\frac{DT_i}{Dt} = \alpha_T \sum_j m_j \frac{T_{ij}}{\rho_i \rho_j} \nabla^2 W_{ij} + h_{\text{conv}}\,(T_{\text{body}} - T_i)\,\phi(r)$$

**Parameters:**

| Symbol | Value | Description |
|--------|-------|-------------|
| $h$ | 0.025 m | Smoothing length |
| $\rho_0$ | 1000 kg/m^3 | Rest density |
| $\gamma$ | 7 | Tait EOS exponent |
| $c_0$ | 10 m/s | Numerical speed of sound |
| $\mu$ | 8 Pa*s | Dynamic viscosity |
| $\mathbf{g}$ | $(0, -9.81)$ m/s^2 | Gravity |
| $dt$ | 0.001 s | Time step |
| $\alpha_T$ | 0.005 m^2/s | Thermal diffusivity |
| $h_{\text{conv}}$ | 8 s^-1 | Convection coefficient (rigid bodies) |
| Domain | $[0,1] \times [0,0.7]$ m | Simulation box |

---

## SPH Kernels

### Cubic Spline (density)

$$W(r, h) = \frac{10}{7\pi h^2} \begin{cases} 1 - \tfrac{3}{2}q^2 + \tfrac{3}{4}q^3 & 0 \le q \le 1 \\ \tfrac{1}{4}(2-q)^3 & 1 < q \le 2 \\ 0 & q > 2 \end{cases}, \qquad q = r/h$$

### Spiky Kernel (pressure gradient)

$$\frac{\partial W_{\text{spiky}}}{\partial r}(r, H) = -\frac{15}{16\pi H^5}(H - r)^2, \qquad H = 2h$$

### Morris Laplacian (viscosity, heat)

$$\nabla^2 A_i \approx 2\sum_j \frac{m_j}{\rho_j} \frac{A_i - A_j}{r_{ij}^2 + \varepsilon^2} \left(r_{ij}\,\frac{\partial W}{\partial r}\right)$$

---

## Equation of State

Pressure follows the **Tait equation** with bulk modulus $B = \rho_0 c_0^2 / \gamma$:

$$p_i = B\left[\left(\frac{\rho_i}{\rho_0}\right)^\gamma - 1\right]$$

This keeps the flow nearly incompressible ($\text{Ma} < 0.1$ by construction) without solving a Poisson equation.

---

## Neighbour Search -- `CellList2D`

Neighbour queries are $O(1)$ amortized using the `num::CellList2D` spatial hash from `include/spatial/cell_list.hpp`:

1. **Build** -- counting-sort all particles into cells of side $2h$ in $O(n + C)$ time ($C$ = cell count).
2. **Sequential backend** -- `grid.iterate_pairs(callback)` visits each pair once (Newton's 3rd law), halving force evaluations.
3. **OpenMP backend** -- per-particle outer loop; each particle queries its $5 \times 5$ cell neighbourhood independently.

---

## Time Integration

Explicit Euler with Morris velocity smoothing:

```
rho_i  <- SPH density sum
p_i    <- Tait EOS
a_i    <- pressure + viscosity + gravity forces
dT     <- SPH heat + rigid-body convection
ev_i   <- (ev_i + v_i) / 2          <- smoothed velocity for viscosity
v_i    += a_i * dt
x_i    += v_i * dt
T_i    += dT/dt * dt
```

Wall collisions apply velocity restitution $e = 0.01$.

---

## Numerics Library Integration

| Feature | Where used |
|---------|-----------|
| `num::SPHKernel<2>` (`spatial/sph_kernel.hpp`) | Cubic spline $W(r,h)$, $dW/dr$; Spiky $dW/dr$ and $\nabla W$ for pressure and viscosity |
| `num::CellList2D` | $O(1)$ neighbour queries; `iterate_pairs` for Newton-3rd-law pair traversal |
| `num::Backend` dispatch | `seq` (pair loop) <-> `omp` (per-particle) selectable at runtime |
| `num::Vector` | Flat particle attribute arrays in the physics solver |

`SPHKernel<2>::W` provides the 2D cubic spline normalisation $10/(7\pi h^2)$; `SPHKernel<2>::Spiky_gradW` returns the pressure-gradient kernel as `std::array<float,2>`, used directly via `auto [gx, gy] = K::Spiky_gradW({rx, ry}, r, h);` in the backends.

---

## Project layout

    include/fluid_sim/sim.hpp   -- consolidated include (FluidSolver, HeatSolver, Particle, Kernel)
    src/fluid.cpp               -- FluidSolver backend dispatch
    src/heat.cpp                -- HeatSolver backend dispatch
    backends/seq/              -- sequential implementations
    backends/omp/              -- OpenMP implementations
    main.cpp                   -- batch frame exporter (dam break preset, 600 frames)
    make_video.sh              -- ffmpeg compositor -> fluid_sim.mp4

---

## Running

```bash
cmake -B build -DNUMERICS_BUILD_FLUID_SIM=ON
cmake --build build --target fluid_sim
./build/apps/fluid_sim/fluid_sim
bash apps/fluid_sim/make_video.sh
```

---

## References

- J. J. Monaghan, *Smoothed Particle Hydrodynamics*, Rep. Prog. Phys. **68** (2005)
- J. P. Morris et al., *Modeling low Reynolds number incompressible flows using SPH*, J. Comput. Phys. **136** (1997) -- viscosity Laplacian
- M. Desbrun & M.-P. Gascuel, *Smoothed Particles: A new paradigm for animating highly deformable bodies*, EGCAS (1996)
