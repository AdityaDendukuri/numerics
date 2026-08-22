# Getting Started {#page_getting_started}

## Add Numerics to a Target

```cmake
find_package(numerics REQUIRED)
target_link_libraries(my_program PRIVATE numerics::numerics)
```

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

## Continue Learning

- @ref page_examples "Browse the code-first feature examples."
- @ref page_solver_best_practices "Choose an appropriate linear solver."
- @ref page_reference "Look up individual API declarations."
