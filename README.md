# numerics

`numerics` is a C++ library I use to develop and test numerical methods for my research.
It grew from code I had written for different projects and courses into a library I
can reuse. I maintain it as my research develops, and some parts are more mature than
others.

The [`examples/`](examples/) show how to use the library. The
[documentation](https://adityadendukuri.github.io/numerics/) has more detail.

## Setup

```cmake
include(FetchContent)
FetchContent_Declare(numerics
    GIT_REPOSITORY https://github.com/AdityaDendukuri/numerics.git
    GIT_TAG v1.1.0)
FetchContent_MakeAvailable(numerics)
target_link_libraries(my_program PRIVATE numerics::numerics)
```

`numerics::core` has no external dependency, and `numerics::kernel` is `num::kernel` alone.

SuiteSparse UMFPACK is optional and uses the GPL. Enable it with
`-DNUMERICS_USE_UMFPACK=ON` and see `THIRD_PARTY_LICENSES.md`.

```bash
cmake --preset dev && cmake --build --preset dev && ctest --preset dev
```

## Examples from my PhD work

- [Refactor only a changed suffix](examples/16_factor_reuse.cpp) of a block matrix, reusing the unchanged prefix.
- [Solve a diffusion problem](examples/15_diffusion_evidence_cg.cpp) with conjugate gradients.
- [Estimate an inverse diagonal](examples/19_probed_inverse_diagonal.cpp) with random probes.
- [Compare two ways to evolve a Markov model](examples/13_talbot_spectral_validation.cpp) using contour inversion and a Krylov method.
- [Compute matrix exponential gradients with Krylov methods](https://github.com/AdityaDendukuri/krylov-gradients) in a separate research library built on `numerics`.

## License

MIT. See `LICENSE` and `THIRD_PARTY_LICENSES.md`.
