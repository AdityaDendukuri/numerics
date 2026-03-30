<!--! @page page_app_tdse 2D Time-Dependent Schrodinger Equation -->

# 2D Time-Dependent Schrodinger Equation

Batch renderer for a quantum wavepacket evolving under the 2D TDSE with five interchangeable potentials. Uses Strang operator splitting with Crank-Nicolson sub-steps, the Thomas algorithm for $O(N)$ tridiagonal solves, and Lanczos eigendecomposition for stationary-state computation.

---

## Preset scene

Double-slit interference -- Gaussian wavepacket (kx=5.0) propagating through a double-slit barrier on a 256x256 grid. Phase-HSV colormap. Exports 800 frames at 30 fps.

---

## Physical Model

The time-dependent Schrodinger equation (in atomic units $\hbar = m = 1$):

$$i\,\frac{\partial \psi}{\partial t} = \hat{H}\psi = \left(-\frac{1}{2}\nabla^2 + V\right)\psi$$

on an $N \times N$ grid over the domain $[0, L]^2$ with Dirichlet boundary conditions $\psi = 0$ on all walls. Interior grid spacing $h = L/(N+1)$.

**Parameters:**

| Symbol | Value | Description |
|--------|-------|-------------|
| $N$ | 256 | Interior grid points per axis (CLI: 32-512) |
| $L$ | 10 | Domain size (atomic units) |
| $h$ | $L/(N+1)$ | Grid spacing |
| $dt$ | 0.004 | Time step |

---

## Strang Operator Splitting

The Hamiltonian is split as $\hat{H} = \hat{T}_x + \hat{T}_y + \hat{V}$. A full time step advances $\psi$ via the symmetric (2nd-order) Strang sequence:

$$\psi(t+dt) = e^{-i\hat{V}dt/2}\; e^{-i\hat{T}_x dt/2}\; e^{-i\hat{T}_y dt}\; e^{-i\hat{T}_x dt/2}\; e^{-i\hat{V}dt/2}\;\psi(t)$$

**Potential kick** (diagonal, exact):

$$\psi \leftarrow \psi \cdot e^{-i V_{i,j}\,\tau}$$

**Kinetic sweep** (Crank-Nicolson in one direction):

$$(I + i\alpha L_x)\psi^{n+1} = (I - i\alpha L_x)\psi^n, \qquad \alpha = \frac{\tau}{4h^2}$$

where $L_x$ is the 1D discrete Laplacian applied row-by-row. Each row yields an $N \times N$ complex tridiagonal system:

$$-i\alpha\,\psi_{k-1} + (1 + 2i\alpha)\,\psi_k - i\alpha\,\psi_{k+1} = d_k$$

---

## Thomas Algorithm

Each row/column tridiagonal system is solved in $O(N)$ using a precomputed LU factorisation (`num::ComplexTriDiag`). Because every row has the **same** coefficients $a = c = -i\alpha$, $b = 1 + 2i\alpha$, the LU factors are computed **once** and reused for all $N$ row solves per half-step:

$$\tilde{c}_k = \frac{c}{\tilde{b}_{k-1}}, \qquad \tilde{b}_k = b - a\tilde{c}_{k-1}$$

This is implemented via `num::thomas` from `include/factorization/thomas.hpp`. Two factorisations are stored: `td_half_` ($\tau = dt/2$) and `td_full_` ($\tau = dt$).

---

## Potentials

| Name | Potential $V(x,y)$ |
|------|--------------------|
| **Free** | $0$ everywhere |
| **Barrier** | Hard wall at $x = 0.55L$ with single gap |
| **DoubleSlit** | Hard wall with two slits separated by $0.12L$ |
| **Harmonic** | $\tfrac{1}{2}\omega^2 r^2$, $\omega = 1.5$ |
| **CircularWell** | $0$ for $r < R$; $V_0 = 5000$ for $r \ge R$, $R = 0.4L$ |

Hard walls use $V_0 \gg 1$ so that $e^{-iV_0 dt/2} \approx 0$ inside the barrier, enforcing reflection.

---

## Eigenmode Computation -- Lanczos

Lanczos iteration runs on the real symmetric operator

$$H_{ij,kl} = \frac{1}{2h^2}\!\left[4\delta_{ij,kl} - \delta_{i\pm1,j;k,l} - \delta_{i,j\pm1;k,l}\right] + V_{ij}\delta_{ij,kl}$$

applied **matrix-free** (no $N^4$ storage). `num::lanczos` from `include/eigen/lanczos.hpp` builds a $K \times K$ symmetric tridiagonal Krylov matrix ($K \approx 5000$) and extracts the lowest eight Ritz pairs by dense eigendecomposition.

For the **CircularWell** potential the exact energies

$$E_{m,n} = \frac{j_{m,n}^2}{2R^2}$$

are computed from the Bessel function zeros $j_{m,n}$ via `num::brent` (`include/analysis/roots.hpp`), solving $J_m(x) = 0$ on bracketed intervals.

---

## Observables

**Norm** (probability conservation):

$$\|\psi\|^2 = h^2 \sum_{i,j} |\psi_{i,j}|^2$$

computed via `num::gauss_legendre` quadrature on row marginals: the 1D row probabilities are first summed, then integrated along $y$ by a 5-point Gauss-Legendre rule from `include/analysis/quadrature.hpp`.

**Energy expectation:**

$$\langle E \rangle = h^2 \sum_{i,j} \operatorname{Re}\!\left(\psi_{i,j}^*\, [H\psi]_{i,j}\right)$$

evaluated via the same matrix-free Hamiltonian matvec used in Lanczos.

---

## Numerics Library Integration

| Feature | Where used |
|---------|-----------|
| `num::CrankNicolsonADI` (`pde/adi.hpp`) | Prefactors two `ComplexTriDiag` systems on construction; `adi.sweep(psi, x_axis, tau)` applies one fiber sweep |
| `num::thomas` / `num::ComplexTriDiag` | $O(N)$ Thomas algorithm called inside each `CrankNicolsonADI` sweep |
| `num::col_fiber_sweep` / `num::row_fiber_sweep` | Fiber extraction and writeback used internally by `CrankNicolsonADI` |
| `num::lanczos` | Builds Krylov subspace for lowest eigenmodes |
| `num::brent` | Bessel zero-finding for exact CircularWell energies |
| `num::gauss_legendre` | Norm / energy quadrature |
| `num::Vector` (complex) | Wavefunction $\psi$, potential $V$, Krylov vectors |

`CrankNicolsonADI` encapsulates the half-step and full-step tridiagonal factorisations, replacing the explicit `td_half_` / `td_full_` members and `cn_sweep_()` implementation that previously lived in the solver directly.

---

## Visualisation

Two colormaps:

**Phase-amplitude (HSV):** hue encodes $\arg\psi \in [-\pi, \pi]$, value encodes $|\psi|^2 / \max|\psi|^2$.

**Probability density:** black -> cyan -> white hot-map with gamma correction $\gamma = 0.45$ to reveal low-amplitude structure.

Potential walls are overlaid as semi-transparent white regions (30 % alpha).

---

## Project layout

    include/tdse/sim.hpp   -- TDSESolver, EigenMode, Stats, Potential declarations
    src/sim.cpp            -- Strang splitting, Thomas algorithm, Lanczos, Bessel zeros
    main.cpp               -- batch frame exporter (double slit preset, 800 frames)
    make_video.sh          -- ffmpeg compositor -> tdse.mp4

---

## Running

```bash
cmake --preset app-tdse
cmake --build --preset app-tdse
./build/apps/tdse/tdse
bash apps/tdse/make_video.sh
```

---

## Build

### Using CMake presets (recommended)

```bash
cmake --preset app-tdse          # configure: Release, tdse only
cmake --build --preset app-tdse
```

`app-tdse` is one of several single-app presets defined in `CMakePresets.json`. Each preset sets `CMAKE_BUILD_TYPE`, enables exactly one app target, and disables everything else so the build stays fast. Available app presets:

| Preset | App | Description |
|--------|-----|-------------|
| `app-tdse`    | `tdse`         | This app -- 2D TDSE quantum simulation |
| `app-quantum` | `quantum_demo` | Interactive quantum circuit simulator |
| `app-fluid`   | `fluid_sim`    | 2D SPH fluid simulation |
| `app-fluid3d` | `fluid_sim_3d` | 3D SPH fluid simulation |
| `app-em`      | `em_demo`      | Electromagnetic field solver |
| `app-ising`   | `ising_sim`    | Ising model Monte Carlo |
| `app-ns`      | `ns_demo`      | 2D Navier-Stokes stress test |

To build every app at once:

```bash
cmake --preset apps
cmake --build --preset apps
```

### Manual configuration

```bash
cmake -B build -DNUMERICS_BUILD_TDSE=ON
cmake --build build --target tdse
```

---

## References

- M. D. Feit et al., *Solution of the Schrodinger equation by a spectral method*, J. Comput. Phys. **47** (1982) -- operator splitting
- D. Kosloff & R. Kosloff, *A Fourier method solution for the time-dependent Schrodinger equation*, J. Comput. Phys. **52** (1983)
- C. Lanczos, *An iteration method for the solution of the eigenvalue problem*, J. Res. Natl. Bur. Stand. **45** (1950)
