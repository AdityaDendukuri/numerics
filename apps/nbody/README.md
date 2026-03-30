<!--! @page page_app_nbody Gravitational N-Body Simulation -->

# Gravitational N-Body Simulation

Batch renderer for a preset galaxy-collapse scene: 200 random bodies under mutual gravity. Integrator: Velocity Verlet (symplectic). Exports 1200 frames at 30 fps.

---

## Physics

Each body $i$ with mass $m_i$ and position $\mathbf{q}_i$ experiences the gravitational acceleration

$$\ddot{\mathbf{q}}_i = G \sum_{j \ne i} m_j \frac{\mathbf{q}_j - \mathbf{q}_i}{\left(|\mathbf{q}_j - \mathbf{q}_i|^2 + \epsilon^2\right)^{3/2}}$$

where $\epsilon = 10^{-3}$ is a softening length that prevents singularities at close approach.

The total mechanical energy is

$$E = \underbrace{\frac{1}{2}\sum_i m_i |\mathbf{v}_i|^2}_{T} - G\sum_{i < j} \frac{m_i m_j}{\sqrt{|\mathbf{q}_i - \mathbf{q}_j|^2 + \epsilon^2}}$$

---

## Numerics Library Integration

| Feature | Where used |
|---------|-----------|
| `num::ode_verlet` (`ode/ode.hpp`) | Default integrator: 2nd-order velocity Verlet (symplectic). FSAL -- only 1 force eval per step after initialization |
| `num::ode_rk4` (`ode/ode.hpp`) | Optional integrator: classic 4th-order Runge-Kutta. 4 force evals per step; energy drifts secularly |
| `num::AccelFn` | Callback type `void(const Vector& q, Vector& acc)` used for the gravitational acceleration function |
| `num::SymplecticCallback` | Callback type `void(real t, const Vector& q, const Vector& v)` -- the app uses this for the energy drift history |
| `num::Vector` | Flat arrays storing all positions and velocities: `q = [x0,y0, x1,y1, ...]`, `v = [vx0,vy0, ...]` |

### Verlet integration (symplectic)

`NBodySim::step(dt)` calls `num::ode_verlet` for a single step:

```cpp
// AccelFn wraps the N-body gravitational sum with softening
num::AccelFn accel = [this](const num::Vector& pos, num::Vector& acc) {
    make_accel(pos, acc);
};

auto res = num::ode_verlet(accel, q, v, t, t + dt, dt);
q = res.q;  v = res.v;  t = res.t;
```

Velocity Verlet is kick-drift-kick: it preserves the symplectic structure of Hamilton's equations, so the energy error remains bounded (it oscillates) rather than accumulating.

### RK4 integration (non-symplectic)

When `use_verlet = false`, positions and velocities are packed into a single state vector and passed to `ode_rk4`:

```cpp
// y = [q0, q1, ..., v0, v1, ...]
Vector y(4 * nb);
std::copy(q.begin(), q.end(), y.begin());
std::copy(v.begin(), v.end(), y.begin() + 2*nb);

auto rhs = [this](real, const Vector& y, Vector& dy) {
    // dy/dt = [v; a(q)]
    ...
};
auto res = num::ode_rk4(rhs, y, t, t + dt, dt);
// unpack back to q, v
```

RK4 is more accurate per step for smooth solutions, but it is not symplectic: energy drifts secularly over long runs.

---

## Project layout

    include/nbody/sim.hpp   -- NBodySim, Body, Scenario declarations
    src/sim.cpp             -- integrator dispatch and scenario initialisation
    main.cpp                -- batch frame exporter (galaxy preset, 1200 frames)
    make_video.sh           -- ffmpeg compositor -> nbody.mp4

---

## Running

Build with CMake (Release):

    cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target nbody
    cd build/apps/nbody
    ./nbody          # renders frames/frame_NNNN.png  (ESC to cancel)
    cd ../../..
    apps/nbody/make_video.sh   # produces nbody.mp4

---

## References

- W. H. Press et al., *Numerical Recipes*, 3rd ed. -- velocity Verlet derivation
- H. Yoshida, *Construction of higher order symplectic integrators*, Phys. Lett. A **150** (1990) -- 4th-order symplectic coefficients
- A. Chenciner & R. Montgomery, *A remarkable periodic solution of the three-body problem*, Ann. Math. **152** (2000) -- figure-8 choreography
