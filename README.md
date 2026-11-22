# numerics

`numerics` is a C++20 library of numerical methods. It has dense, banded and sparse linear
algebra, Krylov solvers, eigenvalues and the SVD, ODE integrators, PDEs, FFTs, quadrature, root
finding, graphs and sampling. The low-level loops are in `num::kernel`, which works on raw
pointers and needs nothing else. BLAS, LAPACK, OpenMP, CUDA, FFTW and SuiteSparse KLU are used if the
build finds them.

I wrote `numerics` to keep the code from my research and my numerical methods courses in one
place. I maintain it alone and it changes as my research does, so use it with caution!!

Each feature has a complete program in [`examples/`](examples/). The
[documentation](https://adityadendukuri.github.io/numerics/) lists them by feature, with their
output, and has a reference page for every public name.

## Setup

```cmake
include(FetchContent)
FetchContent_Declare(numerics
    GIT_REPOSITORY https://github.com/AdityaDendukuri/numerics.git
    GIT_TAG v1.0.0)
FetchContent_MakeAvailable(numerics)
target_link_libraries(my_program PRIVATE numerics::numerics)
```

`numerics::core` has no dependencies. `numerics::kernel` is just `num::kernel`.

```bash
cmake --preset dev && cmake --build --preset dev && ctest --preset dev
```

## From my PhD research

| Feature | Example |
| :--- | :--- |
| Refactoring after a local change. Block LU and block Cholesky keep every block before the change, and no-pivot LU is corrected by Woodbury while few states differ. | [`16_factor_reuse.cpp`](examples/16_factor_reuse.cpp) |
| Solving with A + PQ<sup>T</sup> from a factorization of A, including transposed solves and the updated diagonal of the inverse | [`17_woodbury_low_rank_update.cpp`](examples/17_woodbury_low_rank_update.cpp) |
| diag(A<sup>-1</sup>) of a sparse M-matrix from a block of random probes | [`19_probed_inverse_diagonal.cpp`](examples/19_probed_inverse_diagonal.cpp) |
| (sI - A)x = b at many shifts from one Hessenberg reduction, and inverse Laplace transforms by Talbot quadrature | [`03_resolvent_and_expv.cpp`](examples/03_resolvent_and_expv.cpp), [`13_talbot_spectral_validation.cpp`](examples/13_talbot_spectral_validation.cpp) |
| The approximate Cholesky factorization of Gao, Kyng and Spielman, as a preconditioner for graph Laplacians | [`21_graphs_and_structures.cpp`](examples/21_graphs_and_structures.cpp) |

## License

MIT. See `LICENSE` and `THIRD_PARTY_LICENSES.md`.
