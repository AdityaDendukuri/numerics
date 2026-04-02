# numerics

A C++ scientific computing library with hardware-aware kernels and efficient simulation apps.
Unified API across linear algebra, iterative solvers, FFT, MCMC, PDE/ODE, and spatial data
structures. Backend dispatch (BLAS, SIMD, OpenMP, CUDA) selects the fastest available path
without changing the call site.

## Why numerics?

`numerics` started off as my attempt to standardize all of the code I had written throughout my research projects and courses. Over time, that snowballed into a much more structured package that, in many ways, functions like a C++ analogue of NumPy/SciPy (but designed around C++20 and built for performance from the ground up).

---

## Two API layers

The library exposes two levels of abstraction. You can use either, or mix them.

### High-level: `num::solve(problem, algorithm)`

Modelled after Julia's SciML ecosystem. The problem struct encodes the mathematical formulation; the algorithm struct encodes the numerical method. The two are decoupled; changing the algorithm requires no edits to the problem definition.

```cpp
// Explicit ODE (Lorenz attractor)
num::solve(
    num::ODEProblem{lorenz, {1.0, 0.0, 0.0}, 0.0, 50.0},
    num::RK45{.rtol = 1e-8, .atol = 1e-10},
    observer);

// Implicit ODE (2D heat equation -- backward Euler + sparse CG)
num::Grid2D       grid{N, h};
num::SparseMatrix A      = num::pde::backward_euler_matrix(grid, coeff);
num::LinearSolver solver = num::make_cg_solver(A);

num::ScalarField2D u(grid, init_val);
num::solve(u, num::BackwardEuler{.solver = solver, .dt = dt, .nstep = nstep});

// MCMC (Ising model -- Metropolis sampling)
double m = num::solve(
    num::MCMCProblem{accept_prob, flip, n_sites},
    num::Metropolis{.equilibration = 2000, .measurements = 500},
    measure, rng);
```

The algorithm is a swappable tag. Change `RK45{}` to `RK4{}` or `Euler{}` without touching the problem; change `BackwardEuler{}` to a future `CrankNicolson{}` the same way.

This layer is the right choice for **batch computations**: parameter sweeps, convergence studies, producing output files.

### Low-level: per-step primitives

Every solver also exposes its building blocks directly. These are the right choice when the time loop is driven externally — by a GUI event loop, a real-time renderer, or a custom orchestration layer.

```cpp
// Single Metropolis sweep (used by interactive Ising app each render frame)
num::markov::metropolis_sweep_prob(n_sites, accept_prob, flip, rng);

// Single implicit step (used by fluid sim each substep)
num::ode::advance(u, solver, {1, dt});

// Explicit ODE lazy range (iterate step-by-step with full control)
for (auto [t, y] : num::rk45(lorenz, y0, params))
    record(t, y);
```

The interactive `apps/` (fluid sim, Ising visualizer, NS demo) use this layer because the GUI owns the loop. `num::solve()` would be the wrong abstraction there — the problem changes each frame as the user moves sliders.

---

## Examples

The `examples/` directory shows all three problem classes through the high-level API:

| Example | Physics | Algorithm |
|---------|---------|-----------|
| `lorenz.cpp` | Lorenz attractor (chaotic ODE) | `RK45` adaptive |
| `heat_demo.cpp` | 2D heat equation (implicit PDE) | `BackwardEuler` + sparse CG |
| `ising_demo.cpp` | Ising model magnetization curve | `Metropolis` MCMC |
| `fft_demo.cpp` | FFT / power spectrum | — |

---

## Quick start

### Clone and build

```bash
git clone https://github.com/AdityaDendukuri/numerics
cmake -B build -DNUMERICS_BUILD_TESTS=ON -DNUMERICS_BUILD_BENCHMARKS=ON
cmake --build build -j$(nproc)
./build/tests/numerics_tests
./build/benchmarks/numerics_bench
```

### Use in your own project with CMake FetchContent

```cmake
include(FetchContent)
FetchContent_Declare(
    numerics
    GIT_REPOSITORY https://github.com/AdityaDendukuri/numerics
    GIT_TAG        main
)
FetchContent_MakeAvailable(numerics)
target_link_libraries(my_app PRIVATE numerics)
```

---

## Status report

The report covers: build environment (compiler, detected backends), test pass/fail counts
per suite, benchmark throughput tables for every module, and throughput-vs-size plots.
Sections whose backend was absent at configure time show placeholders.

**Run it first thing after cloning** to verify which backends are active on your machine
and get a baseline of what performance to expect:

```bash
git clone https://github.com/AdityaDendukuri/numerics
cmake -B build \
  -DNUMERICS_BUILD_TESTS=ON \
  -DNUMERICS_BUILD_BENCHMARKS=ON \
  -DNUMERICS_BUILD_REPORT=ON
cmake --build build -j$(nproc)
cmake --build build --target report
# output/REPORT.md  — open this to see your system's results
# output/plots/*.png — throughput-vs-size curves per module
```

CMake reports detected backends at configure time:

```
-- BLAS:   found (/usr/lib/libopenblas.so)
-- FFTW3:  found (fftw3)
-- OpenMP: found (4.5)
-- SIMD:   NEON (AArch64)
-- CUDA:   not found
```

Install and reconfigure if missing a backend — the benchmarks will automatically pick it
up and the report will show the new numbers alongside the fallback paths.

Every push to `main` also runs the full suite in CI. To read the latest CI report:
1. Go to the repo on GitHub → **Actions** tab
2. Open the latest run on `main`
3. Click the **Report** job → rendered `REPORT.md` appears under **Summary**.

---

## Library modules

| Module | What it provides |
|--------|-----------------|
| `core` | `Vector`, `Matrix`, `SparseMatrix`, `BandedMatrix`; backend dispatch |
| `fields` | `Grid2D`, `Grid3D`, `ScalarField2D/3D`, `VectorField3D` — geometry + field types |
| `factorization` | LU (partial pivoting), QR (Householder), Thomas algorithm |
| `solvers` | CG, matrix-free CG, restarted GMRES, Jacobi, Gauss-Seidel; `LinearSolver` callable type |
| `eigen` | Power / inverse / Rayleigh iteration, Lanczos, dense symmetric Jacobi |
| `svd` | Full SVD, randomized truncated SVD |
| `ode` | Euler, RK4, RK45 (adaptive), Verlet, Yoshida4; lazy-range and high-level integrators |
| `pde` | Stencils (2D/3D Laplacian, periodic/Dirichlet), backward-Euler matrix builder |
| `solve` | `num::solve(problem, algorithm)` dispatcher; `ODEProblem`, `MCMCProblem`, algorithm tags |
| `spectral` | FFT, IFFT, RFFT, IRFFT; FFTW3 and native Cooley-Tukey backends; `FFTPlan` |
| `analysis` | Trapz, Simpson, Gauss-Legendre, adaptive Simpson, Romberg; bisection, Newton, secant, Brent |
| `stats` | `RunningStats` (Welford), `Histogram`, `autocorr_time` |
| `markov` | Metropolis sweep, umbrella sampling; template-based, zero overhead |
| `meshfree` | `CellList2D/3D`, `VerletList` -- particle-based and meshfree spatial structures |
| `parallel` | CUDA and MPI interfaces (optional) |

### Backend dispatch

Every compute-intensive operation accepts an optional `Backend` parameter:

```cpp
num::Matrix A(N, N), B(N, N), C(N, N);
matmul(A, B, C);               // default -- blas if available, else blocked
matmul(A, B, C, num::seq);     // naive triple loop
matmul(A, B, C, num::blocked); // cache-blocked scalar
matmul(A, B, C, num::simd);    // AVX2 / NEON intrinsics
matmul(A, B, C, num::blas);    // cblas_dgemm
matmul(A, B, C, num::omp);     // OpenMP tiled
matmul(A, B, C, num::gpu);     // CUDA kernel
```

`default_backend` is a compile-time constant: `blas` when found, otherwise `blocked`.

| Backend | Implementation | Requires |
|---------|---------------|---------|
| `seq` | Naive C++ loops | Always |
| `blocked` | Cache-blocked scalar | Always |
| `simd` | AVX2+FMA (x86) / NEON (AArch64) | Compiler target |
| `blas` | `cblas_dgemm` / `cblas_dgemv` / `cblas_ddot` | OpenBLAS, MKL, or Accelerate |
| `omp` | `#pragma omp parallel for` | OpenMP |
| `gpu` | CUDA kernels | CUDA Toolkit |

The FFT module has its own `FFTBackend` enum (`seq` = Cooley-Tukey, `fftw` = FFTW3).

---

## Physics applications

The `apps/` directory contains standalone interactive simulations built on the low-level API.
Apps are **not built by default** — enable them individually or all at once.

| App | CMake flag | Description |
|-----|-----------|-------------|
| `apps/fluid_sim` | `NUMERICS_BUILD_FLUID_SIM` | 2D weakly-compressible SPH with heat transport and particle injection |
| `apps/fluid_sim_3d` | `NUMERICS_BUILD_FLUID_SIM_3D` | 3D SPH with opposing hose jets and free-orbit camera |
| `apps/ns_demo` | `NUMERICS_BUILD_NS_DEMO` | 2D incompressible Navier-Stokes, Chorin projection, real-time vorticity |
| `apps/ising` | `NUMERICS_BUILD_ISING` | 2D Ising model -- interactive Metropolis with live temperature/field sliders |
| `apps/ising_nucleation` | `NUMERICS_BUILD_ISING` | Umbrella-sampled nucleation, live nucleus size control |
| `apps/tdse` | `NUMERICS_BUILD_TDSE` | 2D time-dependent Schrodinger equation, Strang splitting, Lanczos eigenmodes |
| `apps/em_demo` | `NUMERICS_BUILD_EM_DEMO` | DC current flow + magnetostatics on a 32^3 voxel grid, matrix-free CG |
| `apps/quantum_demo` | `NUMERICS_BUILD_QUANTUM_DEMO` | Quantum circuit statevector simulator |
| `apps/nbody` | `NUMERICS_BUILD_NBODY` | N-body gravitational dynamics, symplectic Verlet |

```bash
# One app
cmake -B build -DNUMERICS_BUILD_FLUID_SIM=ON
cmake --build build -j$(nproc)

# All apps
cmake -B build -DNUMERICS_BUILD_APPS=ON
cmake --build build -j$(nproc)
```

Apps require raylib (fetched automatically) and on Linux also need X11 headers:
```bash
sudo apt install libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libxext-dev
```

---

## Dependencies

| Dependency | How it arrives |
|------------|---------------|
| GTest | FetchContent (automatic) |
| Google Benchmark | FetchContent (automatic) |
| OpenBLAS / MKL / Accelerate | System install, optional |
| FFTW3 | System install, optional |
| OpenMP | System install, optional |
| CUDA Toolkit | System install, optional |
| MPI | System install, optional |
| raylib 5.0 | FetchContent (automatic, **apps only**) |

```bash
# Ubuntu
sudo apt install libopenblas-dev libfftw3-dev

# macOS (Accelerate is built-in; FFTW via Homebrew)
brew install fftw

# OpenMP on macOS clang
brew install libomp
```

---

## Build options

```bash
# Library only (default)
cmake -B build
cmake --build build -j$(nproc)

# With tests and benchmarks
cmake -B build -DNUMERICS_BUILD_TESTS=ON -DNUMERICS_BUILD_BENCHMARKS=ON

# With report generation
cmake -B build -DNUMERICS_BUILD_TESTS=ON -DNUMERICS_BUILD_BENCHMARKS=ON \
               -DNUMERICS_BUILD_REPORT=ON
cmake --build build --target report

# Disable optional backends
cmake -B build -DNUMERICS_USE_BLAS=OFF
cmake -B build -DNUMERICS_USE_OPENMP=OFF
cmake -B build -DNUMERICS_ENABLE_CUDA=OFF
```

CMake reports detected backends at configure time:
```
-- BLAS:   found (/usr/lib/libopenblas.so)
-- FFTW3:  found (fftw3)
-- OpenMP: found (4.5)
-- SIMD:   NEON (AArch64)
-- CUDA:   not found
```

---

## Project layout

```
include/            Public headers (namespace num::)
  core/             Vector, Matrix, types, policy, backends
  fields/           Grid2D, Grid3D, ScalarField2D/3D, VectorField3D
  linalg/
    solvers/        CG, GMRES, Jacobi, Gauss-Seidel; LinearSolver type
    factorization/  LU, QR, Thomas
    eigen/          Power iteration, Lanczos, dense Jacobi
    svd/            Full and truncated SVD
    sparse/         SparseMatrix (CSR), sparse_matvec
    banded/         BandedMatrix, banded_solve
  ode/              Euler, RK4, RK45, Verlet, Yoshida4; implicit advance
  pde/              Stencils, Laplacian builders, diffusion operators
  solve/            num::solve() dispatcher, ODEProblem, MCMCProblem, algorithm tags
  spectral/         FFT interface and FFTBackend enum
  analysis/         Quadrature, root finding
  stats/            RunningStats, Histogram, autocorr_time
  stochastic/       Metropolis sweep, umbrella sampling
  meshfree/         CellList2D/3D, VerletList (particle/meshfree methods)
  parallel/         CUDA and MPI interfaces
  numerics.hpp      Umbrella include

src/                Implementations
examples/           Batch demos using num::solve() (lorenz, heat, ising, fft)
apps/               Interactive simulations using low-level per-step API
tests/              GTest unit tests
benchmarks/         Google Benchmark suite + gnuplot plots
```

---

## Module dependencies

```
core/          (types, Vector, Matrix, policy)
  |
  +-- fields/        (Grid2D, Grid3D, ScalarField -- geometry + field storage)
  |
  +-- linalg/        (sparse, solvers, factorization, eigen, svd)
  |     |
  |     +-- LinearSolver  (universal callable: A*x = rhs)
  |
  +-- ode/           (explicit: Euler/RK4/RK45/Verlet; implicit: advance)
  |
  +-- pde/           (stencils, matrix builders)   <- needs fields/ + linalg/
  |
  +-- solve/         (num::solve dispatcher)        <- needs ode/ + pde/ + stochastic/
  |
  +-- spectral/      (FFT)
  +-- stochastic/    (Metropolis, umbrella)
  +-- meshfree/      (CellList, VerletList -- particle methods)
  +-- analysis/      (quadrature, roots)
  +-- stats/
  +-- parallel/      (CUDA, MPI -- optional)
```
