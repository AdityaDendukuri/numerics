<!--! @page page_app_ising 2D Ising Model -->

# 2D Ising Model

Batch renderer for the 2D Ising model with two modes: free Metropolis dynamics (classic) and umbrella-sampled nucleation. Reproduces the experiments from `IsingNucleation` (Brendel et al. 2005) using the numerics library for all linear-algebra observables.

---

## Preset scene

Nucleation dynamics -- umbrella sampling at T=0.58 (below Tc), F=0.1, nucleus target size 15. Largest spin-down cluster shown in red. Exports 1000 frames at 30 fps.

---

## Physical Model

The system is an $n \times n$ lattice of binary spins $s_i \in \{-1, +1\}$ governed by the Hamiltonian

$$H = -J \sum_{\langle ij \rangle} s_i s_j + F \sum_i s_i$$

where $J = 1$ is the exchange coupling, $F$ is an external field, and the sum $\langle ij \rangle$ runs over nearest-neighbour pairs with periodic boundary conditions. A field $F > 0$ lowers the energy of spin-down sites, driving nucleation of the $s = -1$ (liquid) phase inside a metastable $s = +1$ (vapour) background.

**Parameters** (matching `IsingNucleation/main.c` exactly):

| Symbol | Value | Description |
|--------|-------|-------------|
| $n$    | 300   | Lattice edge length |
| $J$    | 1.0   | Exchange coupling |
| $\beta$| 0.58  | Inverse temperature ($T \approx 1.724$, below $T_c$) |
| $F$    | 0.0 / 0.1 | Field (classic / nucleation) |
| $T_c$  | 2.26918... | Onsager exact critical temperature |

---

## Metropolis Algorithm

A sweep consists of $n^2$ single-spin flip attempts drawn **uniformly at random** (with replacement), matching the original `mc_step_boltz` exactly. The acceptance probability for flipping spin $s_i$ with neighbour sum $\sigma_i = \sum_j s_j$ is

$$P_{\text{acc}} = \min\!\left(1,\; e^{-\beta \Delta E}\right), \qquad \Delta E = 2J s_i \sigma_i - 2F s_i.$$

**Optimisation -- precomputed Boltzmann table.** Because $s_i \in \{-1,+1\}$ and $\sigma_i \in \{-4,-2,0,2,4\}$ (sum of four $\pm 1$ neighbours), $\Delta E$ takes only 10 discrete values. The table

```
boltz[si][ni]  with  si = (s < 0) ? 0 : 1,  ni = sigma_i/2 + 2
```

is rebuilt whenever $T$ or $F$ changes; no `std::exp` is called inside the sweep loop.

**Precomputed neighbour arrays.** Arrays `up[NN]`, `dn[NN]`, `lt[NN]`, `rt[NN]` cache the four PBC-wrapped index offsets, eliminating per-spin modulo arithmetic.

---

## Nucleation and Umbrella Sampling

In nucleation mode the simulation reproduces `run_sim_nucleation` from IsingNucleation:

1. Start with $F = 0.1$ and a square spin-down seed of area $r$ at the centre (`generate_nucleus`).
2. Each sweep is followed by **per-sweep rejection**: if the largest connected spin-down cluster (nucleus) leaves the window $[w - \Delta, w + \Delta]$, the full lattice state is reverted.
3. Windows start at $w = 15$, $\Delta = 15$ (window\_size = 30), matching `run_sim_nucleation` defaults.

The nucleus size $N$ sampled within each window gives the conditional distribution $P(N \mid w)$. Combining windows via **WHAM** reconstructs the nucleation free-energy barrier $\Delta F(N)$.

### Cluster Detection -- Array-Based BFS

The original code uses a recursive DFS (`largest_cluster` in `src/ising_lattice.c`) which risks stack overflow for large clusters on an $n = 300$ lattice ($n^2 = 90000$ sites). The `ClusterDetector` struct replaces this with an iterative BFS backed by a pre-allocated flat queue of size $n^2$:

```
for each unvisited spin-down site 'start':
    assign cluster id  ->  push to queue_buf[qtail++]
    while qhead < qtail:
        i = queue_buf[qhead++]
        for nb in {up[i], dn[i], lt[i], rt[i]}:
            if id[nb] == -1: id[nb] = cid; queue_buf[qtail++] = nb
```

No heap allocation occurs inside `run()`; the buffers are owned by the struct and reused every call.

---

## Numerics Library Integration

| Feature | Where used |
|---------|-----------|
| `num::PBCLattice2D` (`spatial/pbc_lattice.hpp`) | Precomputed `up/dn/lt/rt` neighbour arrays; eliminates per-spin modulo in the hot sweep loop |
| `num::markov::boltzmann_accept` (`stochastic/boltzmann_table.hpp`) | Computes $\min(1, e^{-\beta\Delta E})$; `make_boltzmann_table` precomputes the 10 discrete delta-E values |
| `num::connected_components` (`spatial/connected_components.hpp`) | BFS cluster labelling for nucleation detection; returns `ClusterResult` with `largest_id` and `largest_size` |
| `num::Vector` | `spins`, `ones`, `nbr_buf` |
| `num::SparseMatrix` + `sparse_matvec` | Neighbour-sum matrix for `energy_per_spin()` (SIMD path) |
| `num::dot` (`Backend::seq`) | Magnetisation $\|m\| = \|\mathbf{1}^\top \mathbf{s}\| / n^2$ and energy |
| `num::newton` (`roots.hpp`) | Mean-field order parameter $m = \tanh(4\beta J m + \beta F)$ |
| `num::RunningStats` (`stats.hpp`) | Online mean $\langle N \rangle$, std-dev $\sigma_N$, $\langle\|m\|\rangle$ per umbrella window |
| `num::Histogram` (`stats.hpp`) | Nucleus-size distribution $P(N \mid w)$ per window (input to WHAM) |

### Mean-Field Solver

`mean_field_m` solves the self-consistency equation using `num::newton`:

```cpp
auto f  = [=](real m){ return std::tanh(4*beta*J*m + beta*F) - m; };
auto df = [=](real m){ real t = std::tanh(...); return 4*beta*J*(1-t*t) - 1; };
auto res = newton(f, df, 0.9);
```

The Newton solver from `include/analysis/roots.hpp` replaces 300 manually unrolled iterations.

### Energy Observable

`energy_per_spin` uses the CSR adjacency matrix `adj` (built once in the constructor) and the SIMD-dispatched `sparse_matvec`:

$$\frac{\langle E \rangle}{n^2} = \frac{-J}{4n^2}\, \mathbf{s}^\top A\,\mathbf{s} - \frac{F}{n^2}\,\mathbf{1}^\top \mathbf{s}$$

where $A$ is the $n^2 \times n^2$ 4-neighbour periodic adjacency matrix.

---

## Rendering

- **Classic mode**: white = $s = +1$, black = $s = -1$
- **Nucleation mode**: white = $s = +1$, red = largest connected spin-down cluster (nucleus), dark grey = isolated spin-down sites

---

## Project layout

    include/ising/sim.hpp   -- IsingLattice<N> template (header-only)
    main.cpp                -- batch frame exporter (nucleation preset, 1000 frames)
    make_video.sh           -- ffmpeg compositor -> ising.mp4

---

## Running

```bash
cmake --preset app-ising
cmake --build --preset app-ising
./build/apps/ising/ising
bash apps/ising/make_video.sh
```

---

## Build

```bash
cmake --preset app-ising
cmake --build --preset app-ising
```

---

## References

- L. Onsager, *Crystal Statistics I*, Phys. Rev. **65** (1944) -- exact $T_c$
- K. Brendel et al., *Nucleation in 2D Ising ferromagnets*, Phys. Rev. E **71** (2005)
- A. M. Ferrenberg & R. H. Swendsen, *Optimized Monte Carlo data analysis*, Phys. Rev. Lett. **63** (1989) -- WHAM
