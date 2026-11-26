# numerics

numerics is my personal C++ library for numerical computation. I wrote it for my own research,
and it grew out of code from different projects and courses. I keep it public because it may be
useful to others, especially for teaching: the routines are short and follow the textbook
methods closely.

It is not a rigorously tested library. Most routines have tests, and the parts I use most are
well exercised, but some parts are more developed than others. For production work, use a
library like Eigen, Armadillo or PETSc.

It has routines for linear equations (dense, banded, tridiagonal and sparse, by direct or
iterative methods), least squares, eigenvalues and singular values; for ordinary and partial
differential equations; for Fourier transforms; for roots and integrals; and for statistics,
random sampling and graphs.

It is written in C++20 and is header-only. It uses BLAS, LAPACK, OpenMP, MPI and CUDA when they
are available, but needs none of them: every routine has its own implementation.

## Installation

With CMake, numerics can be fetched as part of a project:

```cmake
include(FetchContent)
FetchContent_Declare(numerics
    GIT_REPOSITORY https://github.com/AdityaDendukuri/numerics.git
    GIT_TAG v1.2.0)
FetchContent_MakeAvailable(numerics)
target_link_libraries(my_program PRIVATE numerics::numerics)
```

A program then includes the whole library with `#include <numerics.hpp>`. The target
`numerics::core` provides only the containers and the basic routines.

## Examples

Each example is a short program that solves one problem.

- [Solve a system of linear equations](reference/examples/01_linear_systems.md).
- [Compute eigenvalues and singular values](reference/examples/04_eigenvalues.md).
- [Integrate an ordinary differential equation](reference/examples/05_differential_equations.md).
- [Solve the heat equation](reference/examples/22_heat_equation.md) by the Fourier transform.
- [Solve a diffusion equation](reference/examples/15_diffusion_equation.md) by the conjugate gradient method.
- [Compute the response of a system of oscillators](reference/examples/23_frequency_response.md) at many frequencies.

## Examples from research

- [Compute how a chemical reaction system evolves](markovkit/index.md#a-first-model), with `markovkit`.
- [Update a factorization](reference/examples/16_updating_a_factorization.md) after part of the matrix changes.
- [Estimate the diagonal of the inverse](reference/examples/19_diagonal_of_the_inverse.md) of a large sparse matrix.
- [Compute a probability distribution in time two ways](reference/examples/13_inverse_laplace_transform.md), and check that they agree.
- [Differentiate the matrix exponential](https://github.com/AdityaDendukuri/krylov-gradients), in a separate library built on numerics.

The [list of examples](reference/examples/index.md) describes every example program, and the
[reference](reference/index.md) describes every routine.
