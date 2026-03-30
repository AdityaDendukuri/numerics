<!--! @page page_app_ns 2D Navier-Stokes Solver -->

# 2D Incompressible Navier-Stokes Solver

Batch renderer for a preset scene: Kelvin-Helmholtz instability -- double shear layer at N=256 with 3000 advected particle tracers. Exports 600 frames at 30 fps.

---

## Physical Model

The 2D incompressible Navier-Stokes equations on a periodic unit square $[0,1]^2$:

$$\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} = -\nabla p + \nu \nabla^2 \mathbf{u}, \qquad \nabla \cdot \mathbf{u} = 0$$

**Parameters:**

| Symbol | Value | Description |
|--------|-------|-------------|
| $N$ | 256 | Grid resolution ($N \times N$ cells) |
| $h$ | $1/N$ | Cell spacing |
| $dt$ | $0.5/N$ | Time step (CFL ~0.5 for semi-Lagrangian advection) |
| $\nu$ | 0.0 | Kinematic viscosity (inviscid Euler by default) |
| Domain | $[0,1]^2$ | Fully periodic |

---

## MAC Grid

Velocity components and pressure are staggered to avoid pressure-velocity decoupling (checkerboard instability):

$$u_{i,j} \;\text{at}\; (ih,\; (j{+}\tfrac12)h), \quad v_{i,j} \;\text{at}\; ((i{+}\tfrac12)h,\; jh), \quad p_{i,j} \;\text{at}\; ((i{+}\tfrac12)h,\; (j{+}\tfrac12)h)$$

All three fields are stored as flat `num::Vector` arrays of length $N^2$.

---

## Chorin Projection Method

Each time step follows three stages:

### 1. Semi-Lagrangian Advection

For each $u$-face at $\mathbf{x} = (ih,\, (j+\tfrac12)h)$, trace the characteristic backward:

$$\mathbf{x}_b = \mathbf{x} - dt\,\mathbf{u}(\mathbf{x})$$

then interpolate bilinearly from surrounding faces:

$$u^\star_{i,j} = \text{interp}_u(\mathbf{x}_b)$$

The departure point $\mathbf{x}_b$ wraps periodically. Semi-Lagrangian advection is unconditionally stable for any $dt$, unlike explicit advection schemes which require $\text{CFL} \le 1$.

### 2. Pressure Poisson Solve

Build the right-hand side from the intermediate divergence:

$$r_{i,j} = -\frac{1}{dt}\,\nabla \cdot \mathbf{u}^\star\big|_{i,j} = -\frac{1}{h\,dt}\left[(u^\star_{i+1,j} - u^\star_{i,j}) + (v^\star_{i,j+1} - v^\star_{i,j})\right]$$

The mean of $r$ is subtracted to remove the constant null-space of the periodic Laplacian. The pressure equation

$$-\nabla^2 p = r$$

is solved with `num::cg_matfree` from `include/solvers/cg.hpp`. The operator $\mathbf{A}\mathbf{p}$ is evaluated matrix-free:

$$(\mathbf{A}\mathbf{p})_{i,j} = \frac{1}{h^2}\left[4p_{i,j} - p_{i-1,j} - p_{i+1,j} - p_{i,j-1} - p_{i,j+1}\right]$$

with periodic index wrapping. The previous pressure field is used as a **warm start**, reducing CG iterations from $O(N)$ to $O(1)$ per step in practice.

### 3. Velocity Projection

Correct the intermediate velocity to enforce $\nabla \cdot \mathbf{u} = 0$:

$$u_{i,j} = u^\star_{i,j} - \frac{dt}{h}(p_{i,j} - p_{i-1,j}), \qquad v_{i,j} = v^\star_{i,j} - \frac{dt}{h}(p_{i,j} - p_{i,j-1})$$

---

## Initial Condition -- Double Shear Layer

The Kelvin-Helmholtz instability is seeded by:

$$u(x,y) = \begin{cases} \tanh\!\left(\tfrac{y-0.25}{\rho}\right) & y \le 0.5 \\ \tanh\!\left(\tfrac{0.75-y}{\rho}\right) & y > 0.5 \end{cases}, \qquad v(x,y) = \delta \sin(2\pi x)$$

with $\rho = 0.05$ (shear layer thickness) and $\delta = 0.05$ (perturbation amplitude). The two shear layers roll up into symmetric vortex streets and merge at late times.

---

## Numerics Library Integration

| Feature | Where used |
|---------|-----------|
| `num::pde::diffusion_step_2d` (`pde/diffusion.hpp`) | Explicit Euler viscosity step: `u* += nu*dt/h^2 * Delta_periodic(u*)` |
| `num::Vector` | Velocity faces $u$, $v$; pressure $p$; intermediates $u^\star$, $v^\star$, rhs |
| `num::cg_matfree` | Matrix-free CG for $-\nabla^2 p = r$; warm-start from previous $p$ |
| `num::dot` (`best_backend`) | Inner products inside CG (BLAS > OMP > blocked) |
| `num::Backend` dispatch | OMP parallelises the per-row matrix-free matvec |
| `num::sample_2d_periodic` | Velocity interpolation for particle tracers and semi-Lagrangian back-trace (bilinear, periodic, configurable stagger offset) |

`diffusion_step_2d` wraps `laplacian_stencil_2d_periodic` + `axpy` into a single call, replacing four lines of scratch-allocation boilerplate. The pressure solve accounts for the majority of wall-clock time. Setting `best_backend` for dot products gives a measurable speedup on multi-core machines.

---

## Visualisation

**Vorticity texture:** The scalar vorticity

$$\omega = \frac{\partial v}{\partial x} - \frac{\partial u}{\partial y}$$

is sampled at every grid corner and mapped to a diverging blue-black-red colormap with adaptive scale (exponential moving average of the per-frame maximum).

**Particle tracers:** 3000 passive particles are advected by the velocity field via `num::sample_2d_periodic`. Tracers are respawned at random positions after 300 frames, tracing the flow topology as white dots.

---

## Project layout

    include/ns_demo/sim.hpp   -- NSSolver declaration
    src/sim.cpp               -- Chorin projection implementation
    main.cpp                  -- batch frame exporter (KH shear layer, 600 frames)
    make_video.sh             -- ffmpeg compositor -> ns_demo.mp4

---

## Running

Build with CMake (Release):

    cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target ns_demo
    cd build/apps/ns_demo
    ./ns_demo          # renders frames/frame_NNNN.png  (ESC to cancel)
    cd ../../..
    apps/ns_demo/make_video.sh   # produces ns_demo.mp4

---

## References

- A. J. Chorin, *Numerical solution of the Navier-Stokes equations*, Math. Comp. **22** (1968)
- R. Bridson, *Fluid Simulation for Computer Graphics*, A K Peters (2008) -- MAC grid, semi-Lagrangian advection
- C. W. Hirt & B. D. Nichols, *Volume of fluid (VOF) method*, J. Comput. Phys. **39** (1981)
