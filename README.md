# numerics

C++ library for numerical linear algebra, ODE/PDE integration, spectral methods, and
statistical simulation. Iterative solvers (CG, GMRES, Lanczos, expv) share a common
kernel layer: BLAS/SIMD primitives, tag-dispatched array operations, and
Modified Gram-Schmidt / Arnoldi routines.

---

## Why numerics?

`numerics` started as an effort to standardize code written across research projects and
coursework. Over time it grew into a structured package that functions like a C++ analogue
of NumPy/SciPy, designed around C++20 and built for performance from the ground up.

---

## Two API layers

### High-level: `num::solve(problem, algorithm)`

Modelled after Julia's SciML ecosystem. The problem struct encodes the mathematical
formulation; the algorithm struct encodes the numerical method. The two are decoupled —
changing the algorithm requires no edits to the problem definition.

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

Change `RK45{}` to `RK4{}` or `Euler{}` without touching the problem definition.
This layer is the right choice for batch computations: parameter sweeps, convergence
studies, producing output files.

### Low-level: per-step primitives

Every solver also exposes its building blocks directly. These are the right choice when
the time loop is driven externally by a GUI event loop, a real-time renderer, or a
custom orchestration layer.

```cpp
// Single Metropolis sweep (used by interactive Ising app each render frame)
num::markov::metropolis_sweep_prob(n_sites, accept_prob, flip, rng);

// Single implicit step (used by fluid sim each substep)
num::ode::advance(u, solver, {1, dt});

// Explicit ODE lazy range (iterate step-by-step with full control)
for (auto [t, y] : num::rk45(lorenz, y0, params))
    record(t, y);
```

---

## Quick start

### Clone and build

```bash
git clone https://github.com/numerics-cpp/numerics
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
    GIT_REPOSITORY https://github.com/numerics-cpp/numerics.git
    GIT_TAG        main
)
FetchContent_MakeAvailable(numerics)
target_link_libraries(my_app PRIVATE numerics)
```

---

## Status report

The report covers: build environment (compiler, detected backends), test pass/fail counts
per suite, benchmark throughput tables for every module, and throughput-vs-size plots.

**Browse the latest report:** [numerics-cpp.github.io/numerics-report](https://numerics-cpp.github.io/numerics-report)

**Generate locally:**

```bash
cmake -B build \
  -DNUMERICS_BUILD_TESTS=ON \
  -DNUMERICS_BUILD_BENCHMARKS=ON \
  -DNUMERICS_BUILD_REPORT=ON
cmake --build build -j$(nproc)
cmake --build build --target report
# output/REPORT.md  -- open this to see your system's results
# output/plots/*.png -- throughput-vs-size curves per module
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

## Library modules

| Module | What it provides |
|--------|-----------------|
| `kernel` | Inner-loop substrate: raw BLAS/SIMD primitives, tag-dispatched array/reduce ops, `LinearOp` interface, MGS orthogonalization, Arnoldi step |
| `core` | `Vector`, `Matrix`, `SparseMatrix`, `BandedMatrix`; high-level backend dispatch for matmul/matvec |
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
| `meshfree` | `CellList2D/3D`, `VerletList` — particle-based and meshfree spatial structures |
| `parallel` | CUDA and MPI interfaces (optional) |

### Kernel module

The `kernel` module is the performance substrate everything else builds on. Three tiers:

**Tier 1 — raw primitives** (`kernel::raw`): always-inline dot, axpy, scale wrappers
that dispatch to BLAS when available, otherwise SIMD/scalar.

**Tier 2 — array and reduce** (`kernel::array`, `kernel::reduce`): tag-dispatched
operations with `seq_t`/`par_t` compile-time policy tags. Zero runtime overhead.

```cpp
ka::axpby(0.5, x, 1.5, y, kernel::kseq);   // sequential
ka::axpby(0.5, x, 1.5, y, kernel::kpar);   // OpenMP parallel
real s = kr::l1_norm(x, kernel::kseq);
```

**Tier 3 — subspace** (`kernel::subspace`): shared inner loops for every Krylov
algorithm. A `LinearOp` interface abstracts the matrix-vector product;
`mgs_orthogonalize` and `arnoldi_step` are the building blocks all solvers call.

```cpp
auto op = kernel::subspace::make_op(
    [&](const Vector& x, Vector& y) { laplacian(x, y, N); }, N * N);

real h_next = kernel::subspace::arnoldi_step(op, V, h, j, scratch);
```

---

## Physics applications

Interactive simulations built on the low-level API live in
[numerics-apps](https://github.com/numerics-cpp/numerics-apps).

| App | Description |
|-----|-------------|
| `fluid_sim` | 2D weakly-compressible SPH with heat transport and particle injection |
| `fluid_sim_3d` | 3D SPH with opposing hose jets and free-orbit camera |
| `ns_demo` | 2D incompressible Navier-Stokes, Chorin projection, real-time vorticity |
| `ising` | 2D Ising model, interactive Metropolis with live temperature/field sliders |
| `tdse` | 2D time-dependent Schrodinger equation, Strang splitting, Lanczos eigenmodes |
| `em_demo` | DC current flow + magnetostatics on a 32^3 voxel grid, matrix-free CG |
| `nbody` | N-body gravitational dynamics, symplectic Verlet |

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

---

## Project layout

```
include/            Public headers (namespace num::)
  kernel/           Performance substrate: raw, array, reduce, subspace tiers
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
tests/              GTest unit tests
benchmarks/         Google Benchmark suite + gnuplot plots
```

---

## Module dependencies

```
kernel/        (raw primitives, array/reduce ops, subspace: MGS, Arnoldi, LinearOp)
  |
core/          (types, Vector, Matrix, policy; matmul/matvec backend dispatch)
  |
  +-- fields/        (Grid2D, Grid3D, ScalarField -- geometry + field storage)
  |
  +-- linalg/        (sparse, solvers, factorization, eigen, svd)
  |     |            uses kernel::subspace for GMRES, Lanczos, expv inner loops
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
