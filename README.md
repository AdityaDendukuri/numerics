# numerics

**Executable mathematics for C++20.**

Numerics is a generic numerical computing library in which mathematical
structure determines which algorithms are valid, evidence-bearing values carry
runtime invariants, and representation-specific operations lower to small,
allocation-controlled kernels.

The intended experience combines three qualities:

- short, familiar calls for ordinary numerical work;
- open generic algorithms that accept foreign types without inheritance;
- explicit mathematical assumptions that remain visible from construction to
  the raw computation.

- [Documentation](https://adityadendukuri.github.io/numerics/)
- [Source](https://github.com/AdityaDendukuri/numerics)

## Start with a solve

```cpp
#include <numerics.hpp>

num::Matrix A(2, 2, 0.0);
A(0, 0) = 4.0;
A(0, 1) = 1.0;
A(1, 0) = 1.0;
A(1, 1) = 3.0;

num::Vector b{1.0, 2.0};
const auto spd_A = num::require<num::axiom::positive_definite>(A);

num::Vector x(2, 0.0);
const auto result = num::cg(spd_A, b, x);
```

`cg` cannot consume an unqualified matrix: its derivation requires a
self-adjoint positive-definite operator. `require` checks that proposition and
returns an immutable, non-owning evidence view. When a construction establishes
the proposition itself, no assertion is needed:

```cpp
const num::operators::BackwardEuler2D system(/*N=*/64, /*dt*diffusivity=*/0.05);
num::Vector rhs(system.rows(), 1.0);
num::Vector u(system.rows(), 0.0);

const auto result = num::cg(system, rhs, u);
```

`BackwardEuler2D` carries positive-definite evidence by construction. See
`examples/15_diffusion_evidence_cg.cpp` for the complete workflow.

## The abstraction model

Numerics separates four things that numerical libraries often conflate:

1. **Protocols** describe executable operations such as `apply`, `inner`, and
   `axpy`.
2. **Type models** explicitly certify algebraic laws such as vector-space,
   inner-product-space, and linear-map structure.
3. **Value evidence** records propositions about a particular object, such as
   positive definiteness or positive definiteness restricted to a subspace.
4. **Representations and capabilities** determine storage and execution without
   changing the mathematical meaning of an algorithm.

An algorithm requires the weakest structure used by its derivation. GMRES needs
a linear endomorphism on an inner-product space; CG additionally needs
positive-definite evidence. Singular graph Laplacians can carry
`positive_definite_on<space::zero_sum>` rather than being mislabeled globally
SPD.

Foreign types participate by supplying operations and opting into laws:

```cpp
namespace num::math {
template <>
struct model_of<MyVector> {
    using laws = type_list<law::inner_product_space>;
};
}
```

No base class or library-owned storage is required.

## Evidence semantics

```cpp
auto stated  = num::assume<num::axiom::positive_definite>(A);
auto checked = num::require<num::axiom::positive_definite>(A);
```

- `assume` records an explicit caller claim while still enforcing decidable
  prerequisites such as square shape.
- `require` runs an available exhaustive validator or fails.
- provenance records how and where evidence was created;
- evidence views expose only `const` access and cannot bind to temporaries.

Evidence views are non-owning. The referenced value must outlive the view and
must not be mutated through another alias while the evidence is in use. Types
that establish invariants by construction avoid this aliasing responsibility.

The older property wrappers remain compatibility carriers while the public API
migrates; new generic algorithms use the `num::math` vocabulary above.

## Architecture

```text
kernel       raw pointers, dimensions, callables, caller-owned workspace
   ^
core/math    spaces, maps, laws, associated types, evidence
   ^
containers + operators + numerical domains
   ^
solve        optional problem-level dispatch
```

The governing rules are:

- raw kernels depend only on the C++ standard library and do not allocate;
- mathematical algorithms are generic over spaces and maps;
- storage fast paths and backend selection live below mathematical algorithms;
- a construction propagates facts it establishes;
- every algorithm has one canonical recurrence;
- optional I/O, plotting, and visualization are leaf facilities.

## Raw kernels

Projects that want only computation can use the standalone kernel directly:

```cpp
#include <kernel/krylov.hpp>

std::vector<double> b(n, 1.0), x(n, 0.0), workspace(3 * n);
auto apply = [&](const double* input, double* output) {
    // output <- A * input
};

const auto result = num::kernel::raw::cg(
    apply, x.data(), b.data(), n, workspace.data(), 1e-10, 500);
```

At this level mathematical preconditions are documented rather than encoded;
buffer validity, dimensions, and workspace ownership belong to the caller.

## Add Numerics with CMake

### FetchContent

```cmake
include(FetchContent)
FetchContent_Declare(
    numerics
    GIT_REPOSITORY https://github.com/AdityaDendukuri/numerics.git
    GIT_TAG main
)
FetchContent_MakeAvailable(numerics)

target_link_libraries(my_program PRIVATE numerics::numerics)
```

### Installed package

```cmake
find_package(numerics REQUIRED)
target_link_libraries(my_program PRIVATE numerics::numerics)
```

The foundational targets can be requested without discovering vendor
dependencies:

```cmake
find_package(numerics REQUIRED COMPONENTS core)
target_link_libraries(my_program PRIVATE numerics::core)
```

Current exported targets:

| Target | Meaning |
| --- | --- |
| `numerics::kernel` | Dependency-free raw computation |
| `numerics::core` | Mathematical protocol and evidence; depends only on `kernel` |
| `numerics::numerics` | Compatibility umbrella with available host capabilities |
| `numerics::blas`, `lapack`, `openmp`, `fftw`, `suitesparse`, `simd` | Named capability targets |
| `numerics::mpi`, `numerics::cuda` | Optional compiled capabilities |

The historical `solvers`, `ode`, `pde`, `spectral`, and `plot` targets currently
forward to the umbrella; they are compatibility aliases, not dependency-isolated
domain packages.

## Build and verify

```bash
cmake --preset dev
cmake --build --preset dev
ctest --preset dev
```

The test suite includes standalone copies of `kernel` and `core`, a package
consumer that requests only `core`, foreign-type interoperability, evidence
provenance and lifetime checks, invariant propagation, convergence tests, and
backend-independent numerical checks.

To build and run the abstraction benchmark:

```bash
cmake -S . -B build/bench -DNUMERICS_BUILD_BENCHMARKS=ON
cmake --build build/bench --target numerics_bench
./build/bench/benchmarks/numerics_bench --benchmark_filter=BM_MathSpine_CG
```

## Project direction

The architectural constitution and migration status live in
`docs/pages/refactor-roadmap.md`. The current priority is to complete one
vertical slice at a time—construction, propagated invariant, generic algorithm,
raw kernel, tests, and benchmark—before introducing broader theory or symbolic
taxonomies.

## License

MIT. See `LICENSE` and `THIRD_PARTY_LICENSES.md`.
