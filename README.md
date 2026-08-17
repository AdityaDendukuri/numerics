# numerics

C++20 library for numerical linear algebra, Krylov methods, spectral transforms, and differential equation solvers.

## Components

### `numerics::kernel`

Core data structures and operators.

* `Vector`
* `Matrix`
* `SparseMatrix`
* `BandedMatrix`
* field types
* `LinearOperator`

### `numerics::solvers`

Linear algebra and Krylov methods.

* LU, QR, and Cholesky
* SVD and symmetric eigensolvers
* CG and GMRES
* Arnoldi matrix exponential actions
* complex resolvent solves

### `numerics::spectral`

Spectral transforms.

* real and complex FFTs
* multidimensional transforms

### `numerics::ode`

ODE integrators.

* RK4
* RK45
* Verlet
* Yoshida4

### `numerics::pde`

Structured PDE solvers and utilities.

* field solvers
* Poisson solvers
* DST-based methods

## Backends

Optional optimized backends are detected at configuration time.

* BLAS
* LAPACK
* OpenMP
* FFTW3
* SIMD

macOS Accelerate and standard Linux BLAS/LAPACK implementations are supported.

## Examples

### Dense and sparse matrices

```cpp
#include <numerics.hpp>

num::Vector x{1.0, 2.0, 3.0};

num::Matrix A(3, 3, 0.0);
A(0, 0) = 4.0;
A(0, 1) = 1.0;
A(1, 0) = 1.0;
A(1, 1) = 4.0;
A(1, 2) = 1.0;
A(2, 1) = 1.0;
A(2, 2) = 4.0;

num::SparseMatrix S(100, 100);
S.insert(0, 0, 2.0);
S.finalize();
```

### Resolvent solve

Solve

$$
(sI-A)x=b.
$$

```cpp
#include <numerics.hpp>

num::Matrix A(2, 2, 0.0);
A(0, 0) = 1.0;
A(0, 1) = 2.0;
A(1, 0) = 3.0;
A(1, 1) = 4.0;

num::Vector b{1.0, 2.0};
num::cplx s(2.0, 1.0);

auto x = num::resolvent_solve(s, A, b);

std::vector<num::cplx> shifts = {
    num::cplx(1.0, 0.0),
    num::cplx(2.0, 1.0),
    num::cplx(0.0, 3.0)
};

auto X = num::resolvent_solve_batch(shifts, A, b);
```

### Matrix exponential action

Compute

$$
e^{tA}v
$$

using an Arnoldi Krylov projection.

```cpp
#include <numerics.hpp>

num::operators::DenseOp Aop(A);
num::Vector v{1.0, 0.0, 0.0};

num::Vector y =
    num::expv(1.0, Aop, v, 30, 1e-8);
```

## Requirements

* C++20
* CMake 3.20+
* BLAS/LAPACK optional
* OpenMP optional
* FFTW3 optional

Supported platforms are macOS and Linux.

## Build

```bash
git clone https://github.com/AdityaDendukuri/numerics.git
cd numerics

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

## License

MIT License. See `LICENSE`.
