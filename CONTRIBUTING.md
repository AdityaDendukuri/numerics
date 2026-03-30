# Contributing to numerics

This is a C++17 codebase written in a deliberately restricted subset of the language.
The goal is code that any C developer can read and any C++ developer can contribute to
without prior knowledge of the project's idioms.

---

## The language subset

### What is allowed
- `struct` for data layout (no member functions, no constructors)
- Free functions operating on structs
- `std::vector`, `std::unordered_map`, `std::unordered_set`, `std::array`
- Namespaces (`num::`, `physics::`) — flat, never nested
- `const` — use it everywhere applicable
- `#pragma once` in every header
- `enum class` for tagged state
- `constexpr` for compile-time constants
- Range-for loops: `for (const auto& p : particles)`

### What is banned
| Feature | Why | Alternative |
|---|---|---|
| `class` | hides data, encourages monolithic design | `struct` + free functions |
| Member functions on structs | same reason | `physics_fluid_step(FluidSolver* s)` |
| Operator overloading | hides function calls | named functions: `vec_add(...)` |
| Exceptions | invisible control flow | return `bool` or a `Result` struct |
| Templates (writing new ones) | complexity, hard to read errors | concrete types or function pointers |
| `auto` outside range-for | hides types at declaration site | write the type explicitly |
| Lambdas with captures | hidden state, unclear lifetimes | named static functions |
| `using namespace` in headers | pollutes every file that includes | always qualify: `std::vector` |
| Nested namespaces | silent symbol hijacking via ADL | one flat namespace per module |
| `static` inside functions | persistent locals are hidden state | pass state explicitly as a parameter |

### `static` — one meaning only
`static` at file scope means internal linkage (not visible outside the `.cpp`). That is
the only use. Mark it with a comment so the intent is clear:

```cpp
static float compute_pressure(float rho) { ... } // internal
```

### `const` — west const, always
```cpp
// correct
const float* ptr;
const FluidParams& p;

// banned
float const* ptr;
```

### `auto` — range-for only
```cpp
// allowed
for (const auto& p : particles) { ... }

// banned — what type is x?
auto x = compute_density(p);

// correct
float x = compute_density(p);
```

---

## Data and functions

Structs hold data. Free functions do work. Keep them separate.

```cpp
// data
struct FluidParams {
    float h    = 0.025f;
    float rho0 = 1000.0f;
    float dt   = 0.001f;
};

// operations
void fluid_step(FluidSolver* s);
void fluid_add_particle(FluidSolver* s, float x, float y, float temp);
void fluid_clear(FluidSolver* s);
```

Functions that mutate state take a pointer. Functions that only read take a const reference.

```cpp
void fluid_step(FluidSolver* s);                    // mutates
float fluid_min_temp(const FluidSolver& s);         // read-only
```

---

## Types

| Use | Type |
|---|---|
| Linear algebra scalars | `num::real` (`double`) |
| Array indices, sizes | `num::idx` (`size_t`) |
| Physics quantities | `float` (performance) |
| Compile-time integer constants | `constexpr int` |

Do not use `long`, `long long`, `uint32_t` etc. unless interfacing with an external API
that requires it. Prefer the aliases above so type width decisions stay in one place.

---

## Naming

| Thing | Convention | Example |
|---|---|---|
| Functions | `snake_case` | `compute_density` |
| Structs | `PascalCase` | `FluidParams` |
| Variables | `snake_case` | `rest_density` |
| Constants | `UPPER_SNAKE` | `MAX_PARTICLES` |
| Namespaces | short lowercase | `num::`, `physics::` |
| Private/internal functions | `snake_case` + `// internal` | see above |

Function names should state what they do to what:
`fluid_step`, `heat_diffuse`, `hash_build`, `hash_query` — not `step`, `diffuse`, `build`.

---

## Headers

- One `#pragma once` at the top of every header, nothing else before it
- Headers declare, `.cpp` files define — keep them in sync
- No implementation in headers except `constexpr` and inline struct definitions
- Include only what the header directly needs — do not rely on transitive includes

```
include/
  linear_algebra/
    core/        types.hpp, vector.hpp, matrix.hpp
    banded/      banded.hpp
    solvers/     solvers.hpp
    grid/        grid3d.hpp
    parallel/    cuda_ops.hpp, mpi_ops.hpp
    linalg.hpp   (umbrella — includes everything above)
  physics/
    fluid/       particle.hpp, rigid_body.hpp, spatial_hash.hpp, kernel.hpp, fluid.hpp, fluid3d.hpp
    heat/        heat.hpp
    electricity/ field.hpp  (ScalarField3D, VectorField3D, FieldSolver, ElectricSolver, MagneticSolver)
    physics.hpp  (umbrella — includes everything above)
  util/
    math.hpp     (constants, Bessel, Legendre, elliptic integrals, linspace, Rng)
src/ mirrors the same layout
```

---

## Error handling

No exceptions. Signal failure via return value.

```cpp
// preferred: bool return, output via pointer
bool solver_step(FluidSolver* s, const char** err_out);

// acceptable for simple cases: assert on preconditions
void hash_build(SpatialHash* h, const Particle* particles, int n) {
    assert(h != nullptr);
    assert(n >= 0);
    ...
}
```

---

## Standard library usage

The C++ standard library is allowed but **raw standard names must not appear in project
code where a wrapper exists**. This keeps the codebase readable without needing to know
the standard's naming choices.

### Approved headers
| Header | Notes |
|---|---|
| `<vector>`, `<unordered_map>`, `<unordered_set>`, `<array>` | Use freely |
| `<cmath>` | Use via `include/linear_algebra/math.hpp` wrappers (see below) |
| `<numeric>` | Use `num::linspace` and `num::seq` instead of `std::iota` directly |
| `<random>` | Use `num::Rng` + `num::rng_*` instead of mt19937 directly |
| `<cassert>` | Use freely for precondition checks |
| `<cstring>`, `<cstdio>` | Allowed for low-level buffer/IO work |
| `<algorithm>` | Allowed; prefer named loops if intent is clearer |

### Math wrappers — `include/linear_algebra/math.hpp`

Always include this header instead of calling `<cmath>` special functions directly.
It provides readable names for everything:

```cpp
#include "linear_algebra/math.hpp"

// constants
num::pi, num::e, num::sqrt2, num::two_pi  ...

// Bessel functions
num::bessel_j(nu, x)      // J_ν(x) — first kind
num::bessel_y(nu, x)      // Y_ν(x) — second kind (Neumann)
num::bessel_i(nu, x)      // I_ν(x) — modified, first kind
num::bessel_k(nu, x)      // K_ν(x) — modified, second kind
num::sph_bessel_j(n, x)   // j_n(x) — spherical, first kind
num::sph_bessel_y(n, x)   // y_n(x) — spherical Neumann

// Orthogonal polynomials
num::legendre(n, x)
num::assoc_legendre(n, m, x)
num::sph_legendre(l, m, theta)
num::hermite(n, x)
num::laguerre(n, x)
num::assoc_laguerre(n, m, x)

// Elliptic integrals
num::ellint_K(k)           // K(k) — complete, first kind
num::ellint_E(k)           // E(k) — complete, second kind
num::ellint_Pi(n, k)       // Π(n,k) — complete, third kind
num::ellint_F(k, phi)      // F(k,φ) — incomplete, first kind

// Other special functions
num::expint(x)             // Ei(x) — exponential integral
num::zeta(x)               // ζ(x) — Riemann zeta
num::beta(a, b)            // B(a,b) — beta function

// Sequence utilities
num::linspace(start, stop, n)   // instead of std::iota for floats
num::seq(start, n)              // integer sequence, wraps std::iota

// Random numbers
num::Rng rng;                       // seeded from hardware entropy
num::Rng rng(42);                   // fixed seed
num::rng_uniform(&rng, 0.0, 1.0);  // instead of mt19937 + distribution boilerplate
num::rng_normal(&rng, 0.0, 1.0);
num::rng_int(&rng, 0, 9);
```

### Not yet available
- `std::mdspan` — C++23. Use `num::Matrix` for 2D arrays; revisit when the project
  moves to C++23.
- `std::numbers::pi` etc. — C++20. Use `num::pi` etc. from `math.hpp`.
- `std::span` — C++20. Pass raw pointer + size for non-owning array views.

---

## Existing code

`num::Vector` and `num::Matrix` in `include/linear_algebra/` are **grandfathered** — they
predate these rules and use `class` with private members. Do not use them as a template
for new code. New linear algebra additions should follow the struct + free function style.

---

## CMake style

Formatting is enforced by `.cmake-format.yaml`. Run before committing:
```bash
pip install cmake-format          # once
cmake-format -i CMakeLists.txt    # format in place
cmake-format --check CMakeLists.txt  # CI check
```

### Rules

**Lowercase commands, uppercase keywords.**
CMake commands are case-insensitive but we always write them lowercase.
`PUBLIC`, `PRIVATE`, `INTERFACE`, `ON`, `OFF`, `BOOL`, `STRING` are always uppercase.

```cmake
# correct
add_library(foo src/foo.cpp)
target_link_libraries(foo PUBLIC bar)

# banned
ADD_LIBRARY(foo src/foo.cpp)
target_link_libraries(foo public bar)
```

**Always use `target_*` commands — never the global forms.**

| Banned (old CMake) | Use instead |
|---|---|
| `include_directories(...)` | `target_include_directories(target PRIVATE ...)` |
| `add_definitions(...)` | `target_compile_definitions(target PRIVATE ...)` |
| `link_libraries(...)` | `target_link_libraries(target PRIVATE ...)` |
| `set(CMAKE_CXX_FLAGS ...)` | `target_compile_options(target PRIVATE ...)` |

**PRIVATE vs PUBLIC vs INTERFACE.**
- `PRIVATE` — dependency is an implementation detail, not exposed to consumers
- `PUBLIC` — dependency is part of the public interface (header includes it)
- `INTERFACE` — header-only: consumers need it, this target does not

```cmake
# src/foo.cpp includes bar.hpp and baz.hpp
# foo.hpp only includes bar.hpp
target_link_libraries(foo
    PUBLIC  bar     # consumers also need bar
    PRIVATE baz     # implementation detail
)
```

**Explicit source lists — never `file(GLOB ...)`.**
`GLOB` silently misses new files until CMake re-runs. List every source explicitly.

```cmake
# correct
add_library(numerics
    src/linear_algebra/core/vector.cpp
    src/linear_algebra/core/matrix.cpp
)

# banned
file(GLOB SOURCES src/*.cpp)
add_library(numerics ${SOURCES})
```

**Option naming — `PROJECTNAME_` prefix, UPPER_SNAKE.**
```cmake
option(NUMERICS_BUILD_TESTS  "Build unit tests"  ON)
option(NUMERICS_ENABLE_CUDA  "Enable CUDA"       OFF)
```

**External dependencies via FetchContent.**
Do not require contributors to pre-install libraries. Use `FetchContent_Declare` +
`FetchContent_MakeAvailable` and pin an exact `GIT_TAG`.

```cmake
FetchContent_Declare(
    raylib
    GIT_REPOSITORY https://github.com/raysan5/raylib.git
    GIT_TAG        5.0
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(raylib)
```

**Section comments.**
Use the `# ===...===` banner style to separate logical sections in top-level
`CMakeLists.txt`. Subdirectory files are short enough to not need them.

---

## Build

Use the named presets. Run `cmake --list-presets` to see all.

| Preset | What it builds | Use when |
|---|---|---|
| `linalg` | Linear algebra lib only | Embedding the solver elsewhere |
| `linalg-test` | linalg + GTest + benchmarks | Working on the math layer |
| `physics` | linalg + physics lib, no raylib | Integrating physics into your own renderer |
| `apps` | Everything including raylib apps | Running the visual simulations |
| `hpc` | linalg + CUDA + MPI | Cluster / supercomputer builds |
| `debug` | Everything, debug symbols | Debugging |
| `ci` | linalg + physics + tests, no CUDA/apps | CI pipelines |

```bash
# configure + build
cmake --preset linalg-test
cmake --build --preset linalg-test

# run tests
ctest --preset linalg-test
```

**Adding a new CMake option:** prefix it `NUMERICS_`, add it to the relevant presets in
`CMakePresets.json`, and document it in the table above.

Tests use GTest (`tests/`). Benchmarks use Google Benchmark (`benchmarks/`).
Add a test for every new function. Benchmarks are optional but welcome for hot paths.
