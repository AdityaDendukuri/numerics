# Getting Started {#page_getting_started}

## Add Numerics to a Target

```cmake
find_package(numerics REQUIRED)
target_link_libraries(my_program PRIVATE numerics::numerics)
```

The default target is the compatibility umbrella and selects host capabilities
found when Numerics was built. Code that needs only the mathematical protocol can
avoid all optional dependency discovery:

```cmake
find_package(numerics REQUIRED COMPONENTS core)
target_link_libraries(my_program PRIVATE numerics::core)
```

Backend capabilities are separate targets (`numerics::blas`,
`numerics::lapack`, `numerics::openmp`, `numerics::fftw`,
`numerics::suitesparse`, and their `numerics::backend::*` aliases). MPI and CUDA
are opt-in compiled targets named `numerics::mpi` and `numerics::cuda`.

Include the complete public API:

```cpp
#include <numerics.hpp> // Import the complete public API.
```

Larger projects can include individual module headers to reduce compile time.

## Define Objects

```cpp
num::Vector x{1.0, 2.0, 3.0}; // Own three contiguous values.
num::Matrix A(3, 3, 0.0);     // Own a zero-filled row-major matrix.

A(0, 0) = 2.0; // Access a matrix entry by row and column.
x[0] = 4.0;     // Access a vector entry by index.
```

## Apply an Operation

```cpp
num::Vector y(3, 0.0); // Allocate caller-owned output.
num::matvec(A, x, y);   // Write y <- A*x.
```

Most low-level operations write into reusable output objects. Factorizations
and high-level solves return objects when ownership is clearer that way.

## Follow a Mathematical Invariant into a Solver

Some constructions establish the conditions required by a downstream
algorithm. A backward-Euler discretization of Dirichlet diffusion constructs a
positive-definite operator, so CG accepts it directly:

```cpp
const num::Grid2D grid{32, 1.0 / 33.0};
const num::operators::BackwardEuler2D system(grid.N, 0.05);

num::Vector rhs(grid.size(), 1.0);
num::Vector solution(grid.size(), 0.0);
const auto result = num::cg(system, rhs, solution);
```

There is no `assume_spd()` call: the operator carries the proposition because
its constructor established it. See @ref 15_diffusion_evidence_cg.cpp for the
complete example.

## Continue Learning

- @ref page_examples "Browse the code-first feature examples."
- @ref page_solver_best_practices "Choose an appropriate linear solver."
- @ref page_reference "Look up individual API declarations."
