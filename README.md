# numerics

`numerics` is a C++20 library of numerical methods. It covers dense, banded and sparse linear
algebra, Krylov solvers, eigenvalues and the SVD, ODE integrators, PDE operators, spectral
transforms, quadrature, root finding, graphs and sampling. The arithmetic lives in `num::kernel`,
a set of allocation-free loops over raw pointers with no dependencies. BLAS, LAPACK, OpenMP,
CUDA, FFTW and SuiteSparse are used when they are present.

I started `numerics` to pull the code from my research and my numerical methods courses
(fluid simulation, stochastic modeling, graph algorithms, linear algebra) into one library. I
maintain it myself, and it keeps growing alongside my research, so use it with appropriate
caution!!

Every feature is explained by a complete program in [`examples/`](examples/), with the output it
prints. The [documentation](https://adityadendukuri.github.io/numerics/) indexes them by feature
and has a reference page for every container, concept and kernel.

## Setup

```cmake
include(FetchContent)
FetchContent_Declare(numerics
    GIT_REPOSITORY https://github.com/AdityaDendukuri/numerics.git
    GIT_TAG v1.0.0)
FetchContent_MakeAvailable(numerics)
target_link_libraries(my_program PRIVATE numerics::numerics)
```

`numerics::core` has no external dependency, and `numerics::kernel` is `num::kernel` alone.

```bash
cmake --preset dev && cmake --build --preset dev && ctest --preset dev
```

## Standout features from my PhD research

| Feature | Example |
| :--- | :--- |
| Refactoring after a local change, with Woodbury-corrected LU while few states differ and block LU that keeps every block before the first change | [`16_factor_reuse.cpp`](examples/16_factor_reuse.cpp) |
| Solving with A + PQ<sup>T</sup> from a factorization of A | [`17_woodbury_low_rank_update.cpp`](examples/17_woodbury_low_rank_update.cpp) |
| diag(A<sup>-1</sup>) of a sparse M-matrix from a block of random probes | [`19_probed_inverse_diagonal.cpp`](examples/19_probed_inverse_diagonal.cpp) |
| Shifted systems (sI - A)x = b at many shifts from one Hessenberg reduction, and Talbot inversion | [`03_resolvent_and_expv.cpp`](examples/03_resolvent_and_expv.cpp), [`13_talbot_spectral_validation.cpp`](examples/13_talbot_spectral_validation.cpp) |

## License

MIT. See `LICENSE` and `THIRD_PARTY_LICENSES.md`.
