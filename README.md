# numerics

`numerics` is a C++20 library for scientific computing.
It provides dense and sparse linear algebra, direct factorizations, Krylov solvers, ODE integrators, spectral transforms, graph algorithms, and quadrature.
The compute kernels are efficient, allocate nothing, and have no dependencies.
The layers above them state the mathematical preconditions of each algorithm, such as symmetry or positive-definiteness, as C++20 concepts.
A precondition that cannot be checked at compile time is asserted by the caller and verified at run time under the diagnostic presets.

I began this library by collecting my research and coursework code in one place, and it still grows in that way.
Tools written for downstream projects are absorbed here and refined for reuse.
The contents range from mesh-free fluid solvers written as an undergraduate for surgical simulation, through graph algorithms and Ising nucleation from my master's work, to finite state projection and iterative linear solvers from my PhD research.
One person maintains it for that research.
Please use it with appropriate caution.

The library has 362 unit tests.
BLAS, LAPACK, OpenMP, and CUDA accelerate it when they are available.
When they are not, portable C++ takes their place.
On one core, the portable dense product and factorizations run within 10% of a vendor BLAS, and the small triangular solves run faster than LAPACK at every size.

Jump right in with the [Documentation](https://adityadendukuri.github.io/numerics/) or browse the [Examples](https://adityadendukuri.github.io/numerics/page_examples.html).

---

## Quickstart

### 1. Direct Factorization and Solve
```cpp
#include <iostream>
#include <numerics.hpp>

int main() {
    num::Matrix A(2, 2, 0.0);
    A(0, 0) = 4.0; A(0, 1) = 1.0;
    A(1, 0) = 1.0; A(1, 1) = 3.0;

    num::Vector b{1.0, 2.0};
    num::Vector x(2, 0.0);

    auto factor = num::cholesky(num::assume_spd(A));
    num::cholesky_solve(factor, b, x); // Solves A * x = b

    std::cout << "x = [" << x[0] << ", " << x[1] << "]\n"; // [0.0909091, 0.636364]
}
```

### 2. Matrix-Free Conjugate Gradient
```cpp
#include <iostream>
#include <numerics.hpp>

int main() {
    constexpr num::idx n = 100;

    // 1D discrete Laplacian: -u''(x)
    auto laplacian = num::operators::make_op(
        [](const num::Vector& u, num::Vector& Lu) {
            const num::idx m = u.size();
            for (num::idx i = 0; i < m; ++i) {
                Lu[i] = 2.0 * u[i] - (i > 0 ? u[i - 1] : 0.0) - (i + 1 < m ? u[i + 1] : 0.0);
            }
        }, n);

    num::Vector b(n, 1.0);
    num::Vector x(n, 0.0);

    auto spd_L = num::operators::assume_spd(laplacian);
    auto res = num::cg(spd_L, b, x, 1e-8, 500);

    std::cout << "CG Converged: " << res.converged << " in " << res.iterations << " iters\n";
}
```

### 3. Adaptive ODE Integration (RK45)
```cpp
#include <iostream>
#include <numerics.hpp>

int main() {
    // Harmonic oscillator: y' = [v, -q]
    auto f = [](double, const num::Vector& y, num::Vector& dy) {
        dy[0] = y[1];
        dy[1] = -y[0];
    };

    num::ODEParams params{.t0 = 0.0, .tf = 10.0, .h = 0.01, .rtol = 1e-8, .atol = 1e-10};
    auto res = num::ode_rk45(f, num::Vector{1.0, 0.0}, params);

    std::cout << "Steps: " << res.steps_taken << ", y(10) = [" << res.y[0] << ", " << res.y[1] << "]\n";
}
```

---

## Structure

```text
kernel       Raw computation (pointers, dimensions, callables; no allocations)
core         Types (idx, real, cplx), backend policy, diagnostics, Models<T, Law>, evidence
algebra      Scalar fields, vector spaces, property hierarchy (spd, self_adjoint, etc.)
container    Vector, Matrix, SparseMatrix, BandedMatrix, SmallMatrix, BLAS/SIMD ops
operator     Matrix-free operators (DenseOp, SparseOp, make_op, projected)
linear       Factorizations (LU, Cholesky, QR, Hessenberg), Krylov (CG, PCG, MINRES, GMRES), SVD/Eigen
ode          IVP integrators (Euler, RK4, RK45, Verlet, Yoshida4)
pde          Stencils, discrete Laplacians, backward Euler, direct Poisson (DST)
spectral     FFT, IFFT, RFFT, IRFFT, FFTPlan, DST-I, 2D DST
spatial      CellList2D, VerletList2D, PBCLattice2D, SPHKernel
structures   DisjointSet, IndexedPriorityQueue, DegreeQueue, Graph, Dijkstra, Kruskal
quadrature   Trapz, Simpson, Gauss-Legendre, Adaptive Simpson, Romberg, Talbot contour
roots        Bisection, Brent, Secant, Newton
stats        RunningStats (Welford), Histogram
stochastic   CategoricalSampler, Metropolis-Hastings, Boltzmann tables, Umbrella sampling
solve        Unified problem dispatch (LinearProblem, ODEProblem)
plot         Terminal ASCII plotting (plt::plot, plt::show_dumb)
```

---

## CMake Integration

### FetchContent
```cmake
include(FetchContent)
FetchContent_Declare(
    numerics
    GIT_REPOSITORY https://github.com/AdityaDendukuri/numerics.git
    GIT_TAG main
)
FetchContent_MakeAvailable(numerics)

target_link_libraries(my_program PRIVATE numerics::numerics)
```

### Installed Package
```cmake
find_package(numerics REQUIRED)
target_link_libraries(my_program PRIVATE numerics::numerics)
```

To link only the dependency-free kernel and mathematical core:
```cmake
find_package(numerics REQUIRED COMPONENTS core)
target_link_libraries(my_program PRIVATE numerics::core)
```

### Exported Targets

| Target | Description |
| :--- | :--- |
| `numerics::kernel` | Standalone raw compute over pointers and callables |
| `numerics::core` | Mathematical protocol and evidence; depends only on `kernel` |
| `numerics::numerics` | Umbrella target with host capability detection |
| `numerics::blas`, `lapack`, `openmp`, `fftw`, `suitesparse`, `simd` | Named capability targets |
| `numerics::mpi`, `numerics::cuda` | Optional compiled capabilities |

---

## Build and Test

```bash
cmake --preset dev
cmake --build --preset dev
ctest --preset dev
```

Run benchmarks:
```bash
cmake -S . -B build/bench -DNUMERICS_BUILD_BENCHMARKS=ON
cmake --build build/bench --target numerics_bench
./build/bench/benchmarks/numerics_bench
```

### LAPACK

The dense factorizations pick between the library's blocked kernel and LAPACK by what
measured faster, not by what is installed:

| Operation | Default |
| :--- | :--- |
| `gemm`, `gemv`, BLAS-1 | vendor BLAS when found |
| Cholesky, triangular solves, LU solves, QR | the kernel, at every size |
| LU factorization | the kernel up to `num::lapack_factor_threshold` (768), LAPACK above |
| SVD, LU inverse | LAPACK when found |

Against OpenBLAS through LAPACKE on an M1 Pro, the kernel's Cholesky was faster at every
size to n = 1536, its LU to about n = 700, and its triangular solves at every size by
2-10x: `dgetrs`/`dpotrs` on a threaded BLAS spend tens of microseconds waking the pool
for an 8 x 8 system, which matters when a simulation performs a million small solves.
The `num::lapack::*` bindings stay callable by name.

LAPACK counts as found only when the library configuration finds one backed by an
optimized BLAS: it reads the shared-object dependencies of the LAPACKE it locates, and
reference LAPACK on reference BLAS (the Homebrew and Debian `lapack` packages) is left
out, since it is slower than the kernel everywhere. Install OpenBLAS (`brew install
openblas`, `apt install libopenblas-dev`) for an optimized LAPACKE, or set
`-DNUMERICS_LAPACK_OPTIMIZED=ON|OFF` to override the check. When benchmarking against
a Homebrew OpenBLAS note that it is built with OpenMP, so its thread count is
`OMP_NUM_THREADS`, not `OPENBLAS_NUM_THREADS`.

---

## License

MIT. See `LICENSE` and `THIRD_PARTY_LICENSES.md`.
