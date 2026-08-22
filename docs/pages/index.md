# numerics {#mainpage}

Numerics is a modular C++20 library for dense and structured linear algebra,
matrix-free solvers, ODE/PDE integration, spectral transforms, and stochastic
methods.

```cpp
#include <numerics.hpp>

num::Matrix A(2, 2, 0.0); // Allocate a 2-by-2 matrix.
A(0, 0) = 4.0;
A(0, 1) = 1.0;
A(1, 0) = 1.0;
A(1, 1) = 3.0;

num::Vector b{1.0, 2.0}; // Define the right-hand side.
auto factor = num::cholesky(num::linalg::make_spd(A));

num::Vector x;
num::cholesky_solve(factor, b, x); // Solve A*x=b.
```

Choose where you want to start:

- @subpage page_getting_started "Getting Started"
- @subpage page_examples "Feature Examples"
- @subpage page_guides "Guides"
- @subpage page_reference "API Reference"
- @subpage page_developer "Developer Documentation"
