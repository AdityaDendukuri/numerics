# pbc_lattice_2d {#page_pbc_lattice}

`include/spatial/pbc_lattice.hpp` provides `num::pbc_lattice_2d`, a small struct that
precomputes the four periodic-boundary neighbor index arrays for an \f$N \times N\f$
square lattice.

---

## Problem

Periodic square-lattice algorithms repeatedly access the four nearest neighbors
of each site.  Precomputed neighbor arrays replace modulo operations inside
sweeps and cluster traversals.

---

## Routine Reference

```cpp
struct num::pbc_lattice_2d {
    int N;
    num::array<int> up, dn, lt, rt;   // N*N each

    explicit pbc_lattice_2d(int N);
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

## Example

```cpp
num::pbc_lattice_2d nbr(N);

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
