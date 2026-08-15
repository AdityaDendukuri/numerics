# numerics

Modular C++20 numerical kernel and solver suite for dense/structured linear algebra, Krylov methods, ODE/PDE integrators, and spectral transforms.

---

## Three Layers & Target Dependencies

```text
                     numerics::kernel  (Layer 1 & 2: Vectors, Matrices, Fields, Operators)
                      /      |      \
                     /       |       \
                    v        |        v
   numerics::spectral        |       numerics::solvers (LU, QR, Cholesky, CG, GMRES, SVD)
                             |        |
                             |        v
                             +---> numerics::ode (RK4, RK45, Verlet, Yoshida4)
                                      |
                                      v
                                  numerics::pde (FieldSolver, Poisson DST-I)
```

| Layer | CMake Target | Components | Recommended Use |
| :--- | :--- | :--- | :--- |
| **Layer 1** | `numerics::raw_kernel` | Header-only raw loops and memory routines | Use for zero-overhead inline memory loops without library compilation. |
| **Layer 2** | `numerics::kernel` | Data structures & operators (`Vector`, `Matrix`, `SparseMatrix`, `BandedMatrix`, `Fields`, `LinearOperator`, `assume_spd()`) | Use when application requires arrays, grids, and operator abstractions without solver overhead or external dependencies. |
| **Layer 3** | `numerics::numerics` | Full solver suite (`solve()`, LU/QR/SVD, CG, GMRES, RK45, PDE, FFT) | Use when complete linear, differential, or spectral solvers are required. |

---

## Code Examples by Layer

### 1. Storage & Core Data Structures (Layer 1 & 2)

```cpp
#include <numerics.hpp>

// Contiguous 1D vector and 2D row-major matrix
num::Vector x{1.0, 2.0, 3.0};
num::Matrix A(3, 3, 0.0);
A(0, 0) = 4.0; A(0, 1) = 1.0;
A(1, 0) = 1.0; A(1, 1) = 4.0; A(1, 2) = 1.0;
A(2, 1) = 1.0; A(2, 2) = 4.0;

// Compressed Sparse Row (CSR) matrix
num::SparseMatrix S(100, 100);
S.insert(0, 0, 2.0);
S.finalize();
```

### 2. Operators & Property Tags (Layer 2)

```cpp
#include <numerics.hpp>

// Dense linear operator wrapper
num::operators::DenseOp Aop(A);
static_assert(num::LinearOperator<decltype(Aop)>);

// SPD property tag (required by CG/PCG solvers)
auto spd_A = num::operators::assume_spd(Aop);
static_assert(num::SPDLinearOperator<decltype(spd_A)>);

// Matrix-free operator y = L(u)
auto Lop = num::operators::make_op(
    [N](const num::Vector& u, num::Vector& Lu) {
        apply_laplacian_2d(u, Lu, N);
    },
    N * N);
```

### 3. Solvers & Numerical Integrators (Layer 3)

```cpp
#include <numerics.hpp>

// Direct LU decomposition
auto fact = num::lu(A);
num::Vector sol;
num::lu_solve(fact, b, sol);

// Iterative Krylov solver
num::LinearSolution s = num::solve(num::LinearProblem{spd_A, b}, num::CG{});

// Adaptive ODE integration (RK45)
auto rhs = [](double t, const num::Vector& y, num::Vector& dy) {
    dy[0] = y[1];
    dy[1] = -y[0];
};
auto ode_res = num::solve(num::ODEProblem{rhs, {1.0, 0.0}, 0.0, 10.0}, num::RK45{});

// Reusable FFT plan
num::spectral::FFTPlan plan(1024, /*forward=*/true);
plan.execute(in, out);
```

---

## C++20 Concept Enforcement

Syntactic interfaces and mathematical properties are verified at compile time.

```cpp
static_assert(num::LinearOperator<decltype(Aop)>);

// Passing untagged operator to CG triggers a static assertion error:
// num::solve(num::LinearProblem{Aop, b}, num::CG{}); // Fails compile-time check

// Wrap with assume_spd() to validate and satisfy SPDLinearOperator concept:
auto spd_A = num::operators::assume_spd(Aop);
num::LinearSolution s = num::solve(num::LinearProblem{spd_A, b}, num::CG{});
```

---

## Runtime Diagnostics

Diagnostic levels inspect dimensions, non-finite values, and mathematical properties at runtime via `std::source_location`.

```cpp
// Set diagnostic depth: off, basic (dims/NaNs), or full (sampled property checks)
num::debug::set_level(num::debug::DiagnosticLevel::full);

// If an operator violates positive definiteness, assume_spd() raises a diagnostic:
// [PropertyError] Error at main.cpp:14 in main:
//   assume_spd() assertion failed: sampled inner product x^T A x = -4.000000 <= 0.
```

---

## Build & Integration

### Build from Source

```bash
cmake -B build -DNUMERICS_BUILD_TESTS=ON
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

### CMake Integration

```cmake
# Layer 1 & 2 (Data structures and operators only):
find_package(numerics REQUIRED)
target_link_libraries(my_app PRIVATE numerics::kernel)

# Layer 3 Component Target (PDE field solvers, transitively links numerics::ode, numerics::solvers, numerics::kernel):
find_package(numerics REQUIRED)
target_link_libraries(my_app PRIVATE numerics::pde)

# Full solver suite:
find_package(numerics REQUIRED)
target_link_libraries(my_app PRIVATE numerics::numerics)
```

---

## Sandbox Script

```bash
./play
```

---

## License

MIT License.
