# PBCLattice2D {#page_pbc_lattice}

`include/spatial/pbc_lattice.hpp` provides `num::PBCLattice2D`, a small struct that
precomputes the four periodic-boundary neighbor index arrays for an \f$N \times N\f$
square lattice.

---

## Motivation

Any 2D simulation on a periodic lattice (Ising, lattice-gas, MD) needs to look up the
four neighbors of each site without calling the modulo operator in the hot path.
This pattern appeared verbatim in the Ising app as a private `build_neighbor_arrays()`
method with four `std::vector<int>` members (`up`, `dn`, `lt`, `rt`).

`PBCLattice2D` extracts the construction and encapsulates the four arrays.

---

## API

```cpp
struct num::PBCLattice2D {
    int N;
    std::vector<int> up, dn, lt, rt;   // N*N each

    explicit PBCLattice2D(int N);
};
```

Construction is \f$O(N^2)\f$ and done once; subsequent lookups are direct array reads.

---

## Index Layout

Flat row-major layout: site \f$(row, col)\f$ has flat index \f$i = row \cdot N + col\f$.

\f[
\texttt{up}[i]  = ((row - 1 + N) \bmod N) \cdot N + col
\f]
\f[
\texttt{dn}[i]  = ((row + 1)     \bmod N) \cdot N + col
\f]
\f[
\texttt{lt}[i]  = row \cdot N + (col - 1 + N) \bmod N
\f]
\f[
\texttt{rt}[i]  = row \cdot N + (col + 1)     \bmod N
\f]

---

## Usage

```cpp
num::PBCLattice2D nbr(N);

// Metropolis sweep -- neighbor sum with no modulo arithmetic
real ns = spins[nbr.up[i]] + spins[nbr.dn[i]]
        + spins[nbr.lt[i]] + spins[nbr.rt[i]];

// BFS cluster detection
num::connected_components(N*N,
    [&](int i) { return spins[i] < 0.0; },
    [&](int i, auto&& visit) {
        visit(nbr.up[i]); visit(nbr.dn[i]);
        visit(nbr.lt[i]); visit(nbr.rt[i]);
    });
```

**Used by:** Ising `IsingLattice::sweep`, `IsingLattice::sweep_umbrella`.
