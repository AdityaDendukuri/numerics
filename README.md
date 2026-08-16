# numerics

[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![CMake](https://img.shields.io/badge/CMake-3.20%2B-brightgreen.svg)](https://cmake.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Linux-lightgrey.svg)]()
[![Tests](https://img.shields.io/badge/Tests-163%2F163%20Passing-success.svg)]()

Modular C++20 numerical kernel and solver suite for **dense/structured linear algebra**, **Krylov subspace solvers**, **resolvent systems**, **ODE/PDE integrators**, and **spectral transforms**.

---

## 🏛️ Architecture & Target Layering

```text
                     numerics::kernel  (Layer 1 & 2: Vectors, Matrices, Fields, Operators)
                      /      |      \
                     /       |       \
                    v        |        v
   numerics::spectral        |       numerics::solvers (LU, QR, Cholesky, CG, GMRES, SVD, Resolvent)
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
| **Layer 3** | `numerics::numerics` | Full solver suite (`solve()`, LU/QR/Cholesky/SVD, CG, GMRES, Resolvent, RK45, PDE, FFT) | Use when complete linear, differential, spectral, or resolvent solvers are required. |

---

## 🚀 Hardware Acceleration Backends

`numerics` features automated compile-time backend dispatch across Linux and macOS:

| Backend | Flag | Supported Operations | macOS Acceleration | Linux Acceleration |
| :--- | :--- | :--- | :--- | :--- |
| **BLAS / cblas** | `NUMERICS_HAS_BLAS` | `dgemm`, `dgemv`, `ddot`, `daxpy`, `dger` | macOS Accelerate | OpenBLAS / BLIS |
| **LAPACK / LAPACKE** | `NUMERICS_HAS_LAPACK` | `dgetrf` (LU), `dgeqrf` (QR), `dpotrf` (Cholesky), `dgesdd` (SVD), `dsyevd` (Eig) | Accelerate C/Fortran Shims | Native `lapacke.h` |
| **OpenMP** | `NUMERICS_HAS_OMP` | Multi-threaded blocked loops, batched resolvents, parallel reductions | AppleClang OpenMP | GCC / Clang libgomp |
| **FFTW3** | `NUMERICS_HAS_FFTW` | 1D/2D/3D Real & Complex DFTs | FFTW3 | FFTW3 |
| **SIMD** | `NUMERICS_HAS_SIMD` | Auto-vectorized array kernels | ARM NEON | AVX2 / AVX-512 |

---

## 💡 Code Examples

### 1. Storage & Core Data Structures
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

### 2. Complex Resolvent Solves: $(s I - A) x = b$
```cpp
#include <numerics.hpp>
#include <iostream>

int main() {
    num::Matrix A(2, 2, 0.0);
    A(0, 0) = 1.0; A(0, 1) = 2.0;
    A(1, 0) = 3.0; A(1, 1) = 4.0;

    num::Vector b{1.0, 2.0};
    num::cplx s(2.0, 1.0);

    // Single shift complex resolvent solve: (sI - A) x = b
    std::vector<num::cplx> x = num::resolvent_solve(s, A, b);

    // Batched shift resolvent solve over OpenMP threads
    std::vector<num::cplx> shifts = {num::cplx(1, 0), num::cplx(2, 1), num::cplx(0, 3)};
    auto batch_sol = num::resolvent_solve_batch(shifts, A, b);

    std::cout << "Resolvent solve completed. x[0] = " << x[0] << "\n";
    return 0;
}
```

### 3. Matrix Exponentials & Arnoldi Krylov Subspace (`num::expv`)
```cpp
#include <numerics.hpp>

// Compute e^{t A} v via m-step Arnoldi Krylov subspace projection
num::operators::DenseOp Aop(A);
num::Vector v{1.0, 0.0, 0.0};
num::Vector exp_tv = num::expv(1.0, Aop, v, 30, 1e-8);
```

---

## 🛠️ Build & Test

```bash
git clone https://github.com/AdityaDendukuri/numerics.git
cd numerics
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for details.
