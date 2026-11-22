# numerics

`numerics` is a C++ library I use to develop and test numerical methods for my research.
It grew from code I had written for different projects and courses into a library I
can reuse. I maintain it as my research develops, and some parts are more mature than
others.

The best way to get started is to run a small example and change it.

## Examples

- [Explore a Hessenberg resolvent](reference/examples/23_hessenberg_frequency_response.md) over many complex shifts.
- [Solve the heat equation with FFT](reference/examples/22_heat_equation_fft.md) and watch a temperature profile smooth out.
- [Solve a linear system](reference/examples/01_direct_factorizations.md) with a small matrix.
- [Evolve a birth and death process](reference/examples/03_resolvent_and_expv.md) over time.
- [Integrate an oscillator](reference/examples/05_symplectic_nbody_ode.md) and compare numerical time steps.
- [Solve a diffusion problem](reference/examples/15_diffusion_evidence_cg.md) with conjugate gradients.

## Examples from my PhD work

- [Watch a reaction model's distribution evolve](markovkit/index.md#a-first-model) with finite state projection in `markovkit`.
- [Refactor only a changed suffix](reference/examples/16_factor_reuse.md) of a block matrix, reusing the unchanged prefix.
- [Estimate an inverse diagonal](reference/examples/19_probed_inverse_diagonal.md) with random probes.
- [Compare two ways to evolve a Markov model](reference/examples/13_talbot_spectral_validation.md) using contour inversion and a Krylov method.
- [Compute matrix exponential gradients with Krylov methods](https://github.com/AdityaDendukuri/krylov-gradients) in a separate research library built on `numerics`.

See [all examples](reference/examples/index.md) or browse the [reference](reference/index.md).

## Setup

```cmake
include(FetchContent)
FetchContent_Declare(numerics
    GIT_REPOSITORY https://github.com/AdityaDendukuri/numerics.git
    GIT_TAG v1.0.0)
FetchContent_MakeAvailable(numerics)
target_link_libraries(my_program PRIVATE numerics::numerics)
```

The full library is available through `numerics::numerics`. Use `numerics::core` for the containers and portable kernels.

`#include <numerics.hpp>` includes the whole library.
