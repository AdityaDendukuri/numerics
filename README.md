# numerics

`numerics` is a modern C++20 numerical computing library for scientific computing, physical simulation, and applied mathematics. It provides cache-aligned dense and sparse linear algebra, direct factorizations, Krylov iterative solvers, adaptive and symplectic ODE integrators, spectral FFT transforms, graph algorithms, and quadrature methods. Compute kernels are designed with zero hidden heap allocations in simulation loops and enforce mathematical preconditions (such as positive-definiteness or symmetry) through C++20 concepts and runtime diagnostic evidence.

This started off as a personal compilation of my research and coursework code into a unified applied math library. I develop this package alongside downstream projects, continuously absorbing and refining new numerical tools into `numerics` for re-use. Because this has primarily been built for my own research workflows rather than by a large team, please use it with appropriate caution!

Over time, this package has grown to include everything from my undergraduate mesh-free fluid solvers (developed for surgical simulation) and master's work on graph algorithms and Ising nucleation, to my PhD research on finite state projection and iterative linear solvers.

Despite its organic evolution, the library is built on modern C++20 with 239 unit tests, clean fallback paths (from BLAS/LAPACK/OpenMP/CUDA acceleration down to pure portable C++).

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

---

## License

MIT. See `LICENSE` and `THIRD_PARTY_LICENSES.md`.
