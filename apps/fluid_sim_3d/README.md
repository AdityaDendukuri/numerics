<!--! @page page_app_fluid3d 3D SPH Fluid Simulation -->

# 3D SPH Fluid Simulation

Batch renderer for three-dimensional Weakly Compressible SPH with two opposing hose jets and coupled heat transport. The solver dispatches to sequential (Newton's-third-law pairs) or OpenMP (per-particle) backends from the numerics library.

Preset scene: dual hose demo -- hot (80 degC) and cold (5 degC) water jets aimed at the centre of a 0.8 m cube. Both hoses always active with a narrow ~2 degree cone. Fixed camera. Exports 500 frames at 30 fps.

---

## Physical Model

Identical governing equations to the 2D fluid sim, extended to $\mathbb{R}^3$. Each particle $i$ carries $(x,y,z)$, $(v_x,v_y,v_z)$, smoothed velocity $(ev_x,ev_y,ev_z)$, $\rho_i$, $p_i$, $T_i$.

**Parameters:**

| Symbol | Value | Description |
|--------|-------|-------------|
| $h$ | 0.05 m | Smoothing length |
| $\rho_0$ | 1000 kg/m^3 | Rest density |
| $\gamma$ | 7 | Tait EOS exponent |
| $c_0$ | 10 m/s | Speed of sound |
| $\mu$ | 10 Pa*s | Dynamic viscosity (higher than 2D for stability) |
| $m$ | $\rho_0 (0.8h)^3$ | Particle mass |
| $dt$ | 0.001 s | Time step |
| $\alpha_T$ | 0.005 m^2/s | Thermal diffusivity |
| $h_{\text{conv}}$ | 8 s^-1 | Convection coefficient |
| Domain | $[0, 0.8]^3$ m | Cubic box |

---

## 3D Kernels

### Cubic Spline

$$W(r, h) = \frac{1}{\pi h^3} \begin{cases} 1 - \tfrac{3}{2}q^2 + \tfrac{3}{4}q^3 & q \le 1 \\ \tfrac{1}{4}(2-q)^3 & 1 < q \le 2 \\ 0 & q > 2 \end{cases}$$

Note the 3D normalisation constant $1/(\pi h^3)$ vs $10/(7\pi h^2)$ in 2D.

### Spiky Kernel (pressure gradient)

$$\frac{\partial W}{\partial r}(r, H) = -\frac{45}{\pi H^6}(H - r)^2, \qquad H = 2h$$

### Morris Laplacian

Identical form to 2D; isotropic in all three spatial directions.

---

## Hose Jets

Two nozzles spray particles with a small random cone offset:

| Hose | Position | Temperature | Direction |
|------|----------|-------------|-----------|
| Cold | $(0.77, 0.76, 0.02)$ m | 5 degC | Toward centre |
| Hot  | $(0.03, 0.76, 0.02)$ m | 80 degC | Toward centre |

Each new particle is given velocity $v_{\text{spray}}\,\hat{\mathbf{d}} + \epsilon_{\text{cone}}$ where $\hat{\mathbf{d}}$ points from nozzle to domain centre and the cone perturbation has half-angle $\pi/90$ rad (~2 degrees).

---

## Neighbour Search -- `CellList3D`

Uses `num::CellList3D` from `include/spatial/cell_list_3d.hpp`:

- Cell size = $2h$ in all three directions.
- **13-stencil forward scan** for Newton-3rd-law pair iteration (half-shell of the 27-cell neighbourhood).
- Build cost $O(n + C_x C_y C_z)$; query cost $O(k)$ per particle.

---

## Numerics Library Integration

| Feature | Where used |
|---------|-----------|
| `num::SPHKernel<3>` (`spatial/sph_kernel.hpp`) | 3D cubic spline $W(r,h)$ with normalisation $1/(\pi h^3)$; Spiky $\nabla W$ returns `std::array<float,3>` |
| `num::CellList3D` | 3D counting-sort spatial hash; Newton-3rd-law pair iteration |
| `num::Backend` dispatch | `seq` (pair traversal) <-> `omp` (per-particle) |
| `num::Vector` | Flat particle attribute arrays |

`SPHKernel<3>` provides the same interface as `SPHKernel<2>` but with the correct 3D normalisations. Gradients are retrieved via `auto [gx, gy, gz] = K::Spiky_gradW({rx, ry, rz}, r, h);` using C++17 structured bindings.

---

## Project layout

    include/fluid_sim_3d/sim.hpp   -- consolidated include (FluidSolver3D, Particle3D, Kernel3D, Heat3D)
    src/sim.cpp                    -- FluidSolver3D backend dispatch
    backends/seq/                 -- sequential implementations
    backends/omp/                 -- OpenMP implementations
    main.cpp                      -- batch frame exporter (hose preset, 500 frames)
    make_video.sh                 -- ffmpeg compositor -> fluid_sim_3d.mp4

---

## Running

```bash
cmake -B build -DNUMERICS_BUILD_FLUID_SIM_3D=ON
cmake --build build --target fluid_sim_3d
./build/apps/fluid_sim_3d/fluid_sim_3d
bash apps/fluid_sim_3d/make_video.sh
```

---

## References

- J. J. Monaghan, *Smoothed Particle Hydrodynamics and its diverse applications*, Annu. Rev. Fluid Mech. **44** (2012)
- M. Muller et al., *Particle-based fluid simulation for interactive applications*, SCA (2003)
