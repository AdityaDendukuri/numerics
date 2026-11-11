# markovkit

`markovkit` is a C++20 library for continuous-time Markov chains and stochastic reaction
networks, built on `numerics`. It solves the chemical master equation by finite state projection,
simulates paths with the SSA, models reaction-diffusion on meshes, and accelerates simulation by
eliminating subnetworks of states.

Source: [github.com/AdityaDendukuri/markovkit](https://github.com/AdityaDendukuri/markovkit)

## Setup

`markovkit` expects `numerics` in the adjacent directory.

```bash
git clone https://github.com/AdityaDendukuri/numerics.git
git clone https://github.com/AdityaDendukuri/markovkit.git
cd markovkit
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
ctest --test-dir build --output-on-failure
```

`markovkit` is header-only. `#include <markovkit.hpp>` includes FSP, SSA and RDME.
`#include <subsweep.hpp>` includes subnetwork sweeping.

## Concepts

Every solver is written against two concepts. A model lists the jumps out of a state; a
state space numbers a finite set of states.

```cpp
template <typename Model>
concept markov_model = requires(const Model &model, const state &x) {
    { model.transitions(x) } -> std::convertible_to<array<transition>>; // {to, rate}
};

template <typename Space>
concept state_space = requires(const Space &space, const state &x, idx i) {
    { space.size() } -> std::convertible_to<idx>;
    { space[i] } -> std::convertible_to<const state &>;
    { space.find(x) } -> std::convertible_to<std::optional<idx>>;
};
```

| Models | Spaces |
| :--- | :--- |
| `reaction_network` | `distribution`: growable, hashed, with probabilities |
| `reaction_diffusion<Graph>` | `population_sector`: fixed species totals, ranked |
| | `population_box`: every sector up to maximum totals |

Any model on any space gives a `generator`, a matrix-free operator for `expv` and Krylov
solvers. `solve_adaptive_fsp` and `solve_krylov_fsp` run the FSP, `gillespie` and
`tau_leaping` sample paths, and `sector_multigrid` preconditions RDME resolvent solves.

## A first model

A birth-death process, simulated exactly and by tau leaping:

```cpp
#include <markovkit.hpp>
using namespace markovkit;

int main() {
    const reaction_network model{{
        {{1}, [](const state &) { return 5.0; }},
        {{-1}, [](const state &x) { return 0.1 * x[0]; }},
    }};

    num::rng random(42);
    const auto exact = gillespie(model, {0}, 20.0, random);
    const auto approximate = tau_leaping(model, {0}, 20.0, random);
}
```

![Gillespie and tau-leaping paths of a birth-death process](ssa_birth_death.png)

The same model gives its distribution by FSP:

```cpp
const auto solution = solve_adaptive_fsp(model, {0}, 2.0, {.inside = box(100)});
solution.means();                        // means(k, s) at solution.times[k]
solution.snapshots.back().marginal(0);
```

An RDME is also a `markov_model`, so SSA and FSP accept it unchanged:

```cpp
const reaction_diffusion rdme{
    voxel_domain(grid<1>({16}, {1.0 / 16}), {1.0}),
    {{{1}, [](view<const int>, real volume) { return 0.05 * volume; }},
     {{-1}, [](view<const int> n, real) { return 0.02 * n[0]; }}}};

const population_box box(rdme.n_voxels(), {3});
const generator Q(box, rdme);
const shifted_generator A(Q, 0.5);                // 0.5 I - Q
const sector_multigrid M(rdme.domain, box, 0.5);
num::gmres(A, M, b, x);
```

## Subnetwork sweeping

A sweep grows a subnetwork of states around the current state, solves for the distribution of
where and when the chain first leaves it, and jumps there. The linear algebra is chosen by
`sweep_settings::regime`:

| Regime | Factorization |
| :--- | :--- |
| `dense` | Retained inverse, initialized by Cholesky for reversible chains and LU otherwise |
| `block` | Block-tridiagonal LU, reusing the factors of unchanged blocks |
| `block_cholesky` | Block Cholesky with the same reuse, for reversible chains |
| `sparse` | Sparse LU, refactored for each subnetwork |
| `automatic` | `block` when the model supplies levels, `sparse` otherwise |

```cpp
sweep_settings settings{.capacity = 120, .depth = 1, .regime = regime::block};
auto walkers = subsweep_walkers(rate_oracle, initial, walker_count, final_time, settings);
```

## Examples

| Program | Shows |
| :--- | :--- |
| `fsp_oregonator` | Adaptive FSP for the Field--Körös--Noyes oscillator |
| `fsp_toggle_switch` | Adaptive FSP for a bistable genetic switch |
| `ssa_birth_death` | Gillespie SSA against tau leaping |
| `rdme_multigrid_benchmark` | Plain and multigrid-preconditioned RDME GMRES scaling |
| `rdme_two_particle_annihilation` | Reaction-time statistics for diffusing `A + B -> 0` |
| `rdme_multigrid_analysis` | Smoothing and two-grid analysis on fixed-population configuration graphs |

![Adaptive FSP for the Oregonator](fsp_oregonator.png)
