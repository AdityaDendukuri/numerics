# numerics

A C++ scientific computing library with hardware-aware kernels and efficient simulation apps.
Unified API across linear algebra, iterative solvers, FFT, MCMC, PDE/ODE, and spatial data
structures. Backend dispatch (BLAS, SIMD, OpenMP, CUDA) selects the fastest available path
without changing the call site.

## Why numerics?

`numerics` started off as my attempt to standardize all of the code I had written throughout my research projects and courses. Over time, that snowballed into a much more structured package that, in many ways, functions like a C++ analogue of NumPy and SciPy.

For example, here is a simple program for simulating the Lorenz attractor:

```cpp
#include "numerics.hpp"

int main() {
    const double sigma = 10.0, rho = 28.0, beta = 8.0 / 3.0;

    auto lorenz = [&](double, const num::Vector& s, num::Vector& ds) {
        ds[0] = sigma * (s[1] - s[0]);
        ds[1] = s[0] * (rho - s[2]) - s[1];
        ds[2] = s[0] * s[1] - beta * s[2];
    };

    num::Vector y0 = {1.0, 0.0, 0.0};
    num::Series xz;

    num::ODEParams params = {.tf = 50.0, .rtol = 1e-8, .atol = 1e-10, .max_steps = 2000000};

    // observer: stores the (x, z) output at each accepted step
    auto observer = [&](double /*t*/, const num::Vector& y) {
        xz.emplace_back(y[0], y[2]);
    };

    // Runge-Kutta RK4(5): t=[0, 50], rtol=1e-8, atol=1e-10
    num::ode_rk45(lorenz, y0, params, observer);

    num::plt::plot(xz);
    num::plt::title("Lorenz attractor (sigma=10, rho=28, beta=8/3)");
    num::plt::xlabel("x");
    num::plt::ylabel("z");
    num::plt::savefig("lorenz.png");
}
```

Another major feature of the library is backend dispatch. Many of the core linear algebra and numerical kernels can automatically route to the fastest available implementation depending on what is installed on your system. That includes support for sequential, SIMD, BLAS, OpenMP, and CUDA backends, while keeping the user-facing call site unchanged.

`numerics` also has a fairly extensive testing and benchmarking framework. Benchmark output can be fed directly into a gnuplot-based workflow to automatically generate plots, which makes it much easier to compare implementations and see where the performance differences actually come from.

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

Every push to `main` runs the full test + benchmark suite in CI and publishes a status
report directly on the Actions run page.

**To view it:**
1. Go to the repo on GitHub -> **Actions** tab
2. Open the latest run on `main`
3. Click the **Report** job
4. The rendered `REPORT.md` appears under **Summary**.

The report covers: build environment (compiler, detected backends), test pass/fail counts
per suite, benchmark throughput tables for every module, and throughput-vs-size plots.
Sections whose backend was absent at configure time show placeholders.

To generate the report locally:

```bash
cmake -B build \
  -DNUMERICS_BUILD_TESTS=ON \
  -DNUMERICS_BUILD_BENCHMARKS=ON \
  -DNUMERICS_BUILD_REPORT=ON
cmake --build build -j$(nproc)
cmake --build build --target report
# output/REPORT.md, output/plots/*.png
```

---

## Library modules

| Module | What it provides |
|--------|-----------------|
| `core` | `Vector`, `Matrix`, `SparseMatrix`, `BandedMatrix`; backend dispatch |
| `factorization` | LU (partial pivoting), QR (Householder), Thomas algorithm |
| `solvers` | CG, matrix-free CG, restarted GMRES, Jacobi, Gauss-Seidel |
| `eigen` | Power / inverse / Rayleigh iteration, Lanczos, dense symmetric Jacobi |
| `svd` | Full SVD, randomized truncated SVD |
| `spectral` | FFT, IFFT, RFFT, IRFFT; FFTW3 and native Cooley-Tukey backends; `FFTPlan` |
| `analysis` | Trapz, Simpson, Gauss-Legendre, adaptive Simpson, Romberg; bisection, Newton, secant, Brent |
| `stats` | `RunningStats` (Welford), `Histogram`, `autocorr_time` |
| `markov` | Metropolis sweep, umbrella sampling; template-based, zero overhead |
| `sparse` | CSR sparse matrix, `sparse_matvec` |
| `banded` | Band-storage matrix, `banded_solve`, `banded_matvec` |
| `spatial` | `CellList2D/3D`, `VerletList` |
| `grid` | `Grid3D` scalar field |
| `parallel` | CUDA and MPI interfaces (optional) |

### Backend dispatch

Every compute-intensive operation accepts an optional `Backend` parameter:

```cpp
#include "numerics.hpp"

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

### Quick example

```cpp
#include "numerics.hpp"
using namespace num;

// Solve Ax = b
SolverResult r = cg(A, b, x);
SolverResult r = cg(A, b, x, 1e-10, 1000, blas);

// Factorizations
LUResult f = lu(A);
lu_solve(f, b, x);

// Eigenvalues
LanczosResult l = lanczos(matvec_fn, n, /*k=*/20);

// SVD
SVDResult s = svd_truncated(A, /*k=*/10);

// FFT
#include "spectral/fft.hpp"
num::CVector X(n);
num::spectral::fft(x, X);                           // default backend
num::spectral::fft(x, X, num::spectral::seq);       // force Cooley-Tukey
num::spectral::FFTPlan plan(n);                     // precomputed, reuse across frames
plan.execute(x, X);

// Quadrature
real I = gauss_legendre([](real x){ return std::sin(x); }, 0.0, pi, 5);
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

## Physics applications

The `apps/` directory contains standalone simulations that use the library.
Apps are **not built by default** -- enable them individually or all at once.

| App | CMake flag | Description |
|-----|-----------|-------------|
| `apps/fluid_sim` | `NUMERICS_BUILD_FLUID_SIM` | 2D weakly-compressible SPH with heat transport and particle injection |
| `apps/fluid_sim_3d` | `NUMERICS_BUILD_FLUID_SIM_3D` | 3D SPH with opposing hose jets and free-orbit camera |
| `apps/ns_demo` | `NUMERICS_BUILD_NS_DEMO` | 2D incompressible Navier-Stokes, Chorin projection, real-time vorticity |
| `apps/ising` | `NUMERICS_BUILD_ISING` | 2D Ising model -- Metropolis dynamics and umbrella-sampled nucleation |
| `apps/tdse` | `NUMERICS_BUILD_TDSE` | 2D time-dependent Schrodinger equation, Strang splitting, Lanczos eigenmodes |
| `apps/em_demo` | `NUMERICS_BUILD_EM_DEMO` | DC current flow + magnetostatics on a 32^3 voxel grid, matrix-free CG |
| `apps/quantum_demo` | `NUMERICS_BUILD_QUANTUM_DEMO` | Quantum circuit demo |

```bash
# One app (use the flag from the table above)
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

## Project layout

```
include/            Public headers (namespace num::)
  core/             Vector, Matrix, types, policy, SmallMatrix
  spectral/         FFT interface and FFTBackend enum
  factorization/    LU, QR, Thomas
  solvers/          CG, GMRES, Jacobi, Gauss-Seidel
  eigen/            Power iteration, Lanczos, dense Jacobi
  svd/              Full and truncated SVD
  analysis/         Quadrature, root finding
  stats/            RunningStats, Histogram, autocorr_time
  markov/           Metropolis, umbrella sampling
  banded/           BandedMatrix, banded_solve
  sparse/           SparseMatrix (CSR), sparse_matvec
  spatial/          CellList2D/3D, VerletList
  grid/             Grid3D
  parallel/         CUDA and MPI interfaces
  numerics.hpp      Umbrella include

src/                Implementations
  core/backends/    seq / blas / omp / gpu / simd
  spectral/         fft.cpp, backends/seq/impl.hpp, backends/fftw/impl.hpp
  ...

apps/               Physics simulations (not built by default)
tests/              GTest unit tests
benchmarks/         Google Benchmark suite + gnuplot plots
report/             Report generator (gen_report.cpp, template.md)
output/             Generated files -- REPORT.md, plots/, JSON  [gitignored]
cmake/              Build helpers
docs/               Doxygen configuration and pages
```

---

## Module dependencies

```
numerics.hpp
  +-- core/          (types, Vector, Matrix, policy)
  +-- spectral/      (FFT, FFTPlan)                  <- depends on core/
  +-- analysis/      (quadrature, roots, stats)      <- no deps on other modules
  +-- markov/        (mcmc, rng)                     <- depends on core/
  +-- factorization/ (lu, qr, thomas)                <- depends on core/
  +-- eigen/         (power, lanczos, jacobi_eig)    <- depends on core/, factorization/
  +-- svd/                                           <- depends on core/, eigen/
  +-- solvers/       (cg, gmres, jacobi, gauss_seidel) <- depends on core/
  +-- sparse/                                        <- depends on core/
  +-- banded/                                        <- depends on core/
  +-- grid/                                          <- depends on core/
  +-- spatial/       (CellList2D/3D, VerletList)     <- depends on core/
  +-- parallel/      (cuda, mpi)                     <- optional, depends on core/
```
