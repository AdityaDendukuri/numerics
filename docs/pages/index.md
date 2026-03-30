# Numerics Library {#mainpage}

A C++ scientific computing library: linear algebra, ODE/PDE solvers, spectral methods, statistical mechanics, and gnuplot-based plotting -- all under a single `#include "numerics.hpp"`.

---

## Example: Lorenz attractor

The Lorenz system is integrated with `num::ode_rk45` (adaptive Dormand-Prince) and the trajectory is plotted with `num::Gnuplot` -- the full program is ~50 lines (`examples/lorenz.cpp`):

```cpp
#include "numerics.hpp"
#include <cstdio>

int main() {
    const double sigma = 10.0, rho = 28.0, beta = 8.0 / 3.0;

    auto lorenz = [&](double, const num::Vector& s, num::Vector& ds) {
        ds[0] = sigma * (s[1] - s[0]);
        ds[1] = s[0] * (rho - s[2]) - s[1];
        ds[2] = s[0] * s[1] - beta * s[2];
    };

    num::Series xz;
    xz.reserve(200000);

    num::Vector y0 = {1.0, 0.0, 0.0};
    auto result = num::ode_rk45(lorenz, y0, 0.0, 50.0,
                                 1e-8, 1e-10, 1e-3, 2000000,
                                 [&](double, const num::Vector& s) {
                                     xz.emplace_back(s[0], s[2]);
                                 });

    printf("%zu steps,  final t = %.4f\n", (size_t)result.steps, result.t);

    num::plt::plot(xz);
    num::plt::title("Lorenz attractor  (sigma=10, rho=28, beta=8/3)");
    num::plt::xlabel("x");
    num::plt::ylabel("z");
    num::plt::show();
}
```

Build and run (requires gnuplot in PATH):

```
g++ -std=c++17 -O2 -Iinclude examples/lorenz.cpp src/ode/ode.cpp -o lorenz
./lorenz
```

This is what the plot generates:

\image html lorenz.png "Lorenz attractor phase portrait (x vs z)" width=600px

---

## Library Modules

### core -- Vectors, matrices, and backend dispatch

- `num::Vector`, `num::Matrix`, `num::SparseMatrix`, `num::BandedMatrix`
- `num::Backend` -- `seq`, `blocked`, `simd`, `blas`, `omp`, `gpu`
- @ref num::matmul, @ref num::matmul_blocked, @ref num::matmul_register_blocked, @ref num::matmul_simd
- @ref num::matvec, @ref num::matadd
- @ref num::dot, @ref num::axpy, @ref num::scale, @ref num::norm
- @subpage page_performance -- cache blocking, SIMD, BLAS, backend selection
- @subpage page_parallel -- OpenMP and CUDA backends

### factorization -- Direct linear solvers

- @subpage page_factorizations
- @ref num::lu, @ref num::lu_solve, @ref num::lu_det, @ref num::lu_inv
- @ref num::qr, @ref num::qr_solve
- @ref num::thomas -- O(n) tridiagonal (Thomas algorithm)

### solvers -- Iterative solvers

- @subpage page_linear_solvers
- @ref num::cg, @ref num::cg_matfree -- conjugate gradient
- @ref num::gmres -- restarted GMRES (Krylov)
- @ref num::jacobi, @ref num::gauss_seidel -- stationary iterative

### eigen -- Eigenvalue methods

- @subpage page_eigenvalues
- @ref num::power_iteration, @ref num::inverse_iteration, @ref num::rayleigh_iteration
- @ref num::lanczos -- Krylov-Lanczos with Ritz extraction
- @ref num::eig_sym -- dense symmetric eigensolver (Jacobi sweeps)

### svd -- Singular value decomposition

- @subpage page_svd
- @ref num::svd -- full SVD
- @ref num::svd_truncated -- randomized truncated SVD

### spectral -- Fourier transforms

- @subpage page_fft
- `num::spectral::fft`, `num::spectral::ifft` -- complex DFT / inverse DFT
- `num::spectral::rfft`, `num::spectral::irfft` -- real-to-complex and inverse
- `num::spectral::FFTPlan` -- precomputed plan for repeated transforms
- `FFTBackend::seq` (Cooley-Tukey radix-2), `FFTBackend::fftw` (FFTW3, optional)

### analysis -- Quadrature and root finding

- @subpage page_analysis
- **Quadrature:** @ref num::trapz, @ref num::simpson, @ref num::gauss_legendre, @ref num::adaptive_simpson, @ref num::romberg
- **Root finding:** @ref num::bisection, @ref num::newton, @ref num::secant, @ref num::brent

### stats -- Simulation observables

- @ref num::RunningStats -- Welford online mean + variance, O(1) memory
- @ref num::Histogram -- fixed-bin histogram; reweighting for WHAM analysis
- @ref num::autocorr_time -- integrated autocorrelation time (Madras-Sokal windowing)

### markov -- Markov chain Monte Carlo

- @ref num::markov::metropolis_sweep -- Metropolis sweep; computes exp(-beta\*dE) internally
- @ref num::markov::metropolis_sweep_prob -- caller-supplied acceptance probability (e.g. Boltzmann table)
- @ref num::markov::umbrella_sweep, @ref num::markov::umbrella_sweep_prob -- umbrella sampling with save/restore
- `num::markov::UmbrellaWindow`, `MetropolisStats`, `UmbrellaStats`
- @ref num::markov::make_seeded_rng -- hardware-entropy RNG seeding

### sparse and banded -- Structured sparse formats

- `num::SparseMatrix` -- CSR format; @ref num::sparse_matvec
- `num::BandedMatrix` -- band-storage; @ref num::banded_matvec, @ref num::banded_solve

### spatial -- Spatial data structures and grid

- @ref num::CellList2D, @ref num::CellList3D -- linked-cell neighbour search
- @ref num::VerletList -- Verlet neighbour list with skin radius
- @ref num::Grid3D -- 3D scalar field with index helpers (Poisson/diffusion interop)

---

### plot -- Gnuplot pipe wrapper

- `num::Gnuplot` -- popen wrapper: `operator<<` for commands, `send1d()` for inline data
- `num::Series` -- `std::vector<std::pair<double,double>>` for (x, y) data
- `num::apply_siam_style(gp)` -- clean SIAM-style theme
- `num::save_png(gp, file, w, h)` -- redirect output to PNG
- `num::set_loglog(gp)`, `num::set_logx(gp)` -- axis scaling helpers

---

## Physics Simulations

Raylib-rendered simulations built on numerics. Each runs as a batch renderer:
simulate all frames, export PNGs, composite with `make_video.sh`.

### Monte Carlo

#### 2D Ising Model

@subpage page_app_ising

Metropolis dynamics and umbrella-sampled nucleation on a 300x300 lattice. Reproduces Brendel et al. (2005) nucleation experiments.

| Library feature | Role |
|-----------------|------|
| `num::markov::metropolis_sweep_prob` | Sweep hot path with precomputed Boltzmann table |
| `num::markov::umbrella_sweep_prob` | Per-sweep rejection / save-restore for window sampling |
| `num::SparseMatrix` + `sparse_matvec` | Neighbour-sum matrix for energy observable (SIMD path) |
| `num::RunningStats`, `num::Histogram` | Online mean, variance, nucleus-size distribution per window |
| `num::newton` | Mean-field self-consistency equation solver |

---

### Fluid Dynamics

#### 2D SPH Fluid Simulation

@subpage page_app_fluid

Weakly compressible SPH with heat transport, rigid bodies, and particle injection. Tait EOS, cubic-spline kernel, Morris viscosity Laplacian.

| Library feature | Role |
|-----------------|------|
| `num::CellList2D` | O(1) neighbour queries; Newton-3rd-law pair traversal |
| `num::Backend` dispatch | `seq` (pair loop) <-> `omp` (per-particle) selectable at runtime |
| `num::Vector` | Flat particle attribute arrays |

#### 3D SPH Fluid Simulation

@subpage page_app_fluid3d

3D WCSPH with opposing hose jets, heat transport, and a free-orbit camera whose orientation rotates the gravity vector.

| Library feature | Role |
|-----------------|------|
| `num::CellList3D` | 3D counting-sort spatial hash; 13-stencil Newton-3rd-law pairs |
| `num::Backend` dispatch | `seq` (pair traversal) <-> `omp` (per-particle) |
| `num::Vector` | Flat particle attribute arrays |

#### 2D Incompressible Navier-Stokes

@subpage page_app_ns

Chorin projection on a staggered MAC grid; semi-Lagrangian advection; matrix-free CG pressure solve. Kelvin-Helmholtz double shear layer initial condition.

| Library feature | Role |
|-----------------|------|
| `num::cg_matfree` | Matrix-free CG for -grad^2p = r; warm-start from previous pressure field |
| `num::dot` | Inner products inside CG (dispatched to best available backend) |
| `num::Vector` | Velocity faces u, v; pressure p; intermediates u*, v*, rhs |

---

### Quantum Mechanics

#### 2D Time-Dependent Schrodinger Equation

@subpage page_app_tdse

Strang operator splitting with Crank-Nicolson kinetic sweeps; Thomas algorithm for O(N) tridiagonal solves; Lanczos eigendecomposition; five interchangeable potentials.

| Library feature | Role |
|-----------------|------|
| `num::thomas` | O(N) complex tridiagonal solve per row/column (kinetic sweeps) |
| `num::lanczos` | Krylov subspace for lowest eigenmodes |
| `num::brent` | Bessel zero-finding for exact circular-well energies |
| `num::gauss_legendre` | Norm and energy quadrature |

---

### Electromagnetism

#### Electromagnetic Field Demo

@subpage page_app_em

DC current flow + magnetostatics on a 32^3 voxel grid. Four Poisson problems (one electric, three magnetic) solved with matrix-free CG; interactive magnetic dipole.

| Library feature | Role |
|-----------------|------|
| `num::cg_matfree` | All four Poisson solves (variable-coefficient electric + 3x magnetic) |
| `num::Grid3D` | 3D scalar field storage for phi, A, J, B components |
| `num::Vector` | Flattened field arrays passed to the solver |

---

## Status Report

Run `cmake --build build --target report` to generate `output/REPORT.md` -- a full snapshot of the current build:

- **Build environment** -- compiler, flags, detected backends (BLAS, FFTW3, OpenMP, CUDA, MPI)
- **Test summary** -- pass/fail counts per suite
- **Benchmark tables** -- throughput and timing for every module (linalg, FFT, banded, analysis)
- **Plots** -- PNG throughput curves (seq vs fftw, size scaling) in `output/plots/`

Sections whose backend was not found at configure time are left as placeholders, so the report always reflects exactly what is installed.

---

## Lecture Notes

- @subpage page_week1
- @subpage page_week2
- @subpage page_week3
- @subpage page_week4
- @subpage page_week5
- @subpage page_week6
- @subpage page_week7
- @subpage page_week8
- @subpage page_week9
