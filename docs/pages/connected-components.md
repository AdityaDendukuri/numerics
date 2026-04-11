# Connected Components {#page_connected_components}

`include/spatial/connected_components.hpp` provides `num::connected_components`,
a template BFS labelling function with a pre-allocated flat queue
(no heap allocation per call, no recursion).

---

## Motivation

The Ising nucleation app needed BFS over spin-down clusters to identify the largest
nucleus for umbrella sampling.  The original `ClusterDetector` struct embedded the
queue, the `id` array, and the BFS logic together -- 45 lines of app-specific code
that is, in fact, a general graph algorithm.

`connected_components` extracts the algorithm into the library with a callable-based
interface that costs nothing at runtime (no `std::function`).

---

## API

```cpp
struct num::ClusterResult {
    std::vector<int> id;        // id[i]: -2 = excluded, >=0 = cluster index
    std::vector<int> sizes;     // sizes[c] = number of sites in cluster c
    int largest_id   = -1;
    int largest_size = 0;
};

template<typename InCluster, typename Neighbors>
num::ClusterResult num::connected_components(
    int n_sites,
    InCluster&&  in_cluster,   // bool(int i)        -- include site i?
    Neighbors&&  neighbors);   // void(int i, F&& f) -- call f(nb) per neighbor
```

### Labels

| `id[i]` | Meaning |
|---|---|
| `-2` | Excluded -- `in_cluster(i)` returned `false` |
| `>=0` | Cluster index |

Sites with `id[i] == ClusterResult::largest_id` belong to the largest cluster.

---

## Algorithm

Iterative BFS with a pre-allocated flat queue of size `n_sites`:

```
for each unvisited included site s:
    assign cluster id, push to queue
    while queue not empty:
        pop i, increment size
        for each neighbor nb of i:
            if unvisited: label nb, push
    update largest_id / largest_size
```

No heap allocations after the initial `ClusterResult` construction.

---

## Usage -- Ising nucleation

```cpp
num::PBCLattice2D nbr(N);
num::ClusterResult det;

// In the Metropolis order-parameter measurement:
det = num::connected_components(N*N,
    [&](int i) { return spins[i] < 0.0; },      // spin-down sites only
    [&](int i, auto&& visit) {
        visit(nbr.up[i]); visit(nbr.dn[i]);
        visit(nbr.lt[i]); visit(nbr.rt[i]);
    });
int nucleus_size = det.largest_size;

// Rendering: highlight the largest cluster red
if (det.id[i] == det.largest_id) { /* draw red */ }
```

**Used by:** Ising `IsingLattice::sweep_umbrella`.

---

## Generalizations

The callable interface accepts any graph topology -- not just 2D lattices:

```cpp
// 3D cubic lattice with PBC
det = num::connected_components(nx*ny*nz,
    [&](int i) { return active[i]; },
    [&](int i, auto&& visit) {
        for (int nb : six_neighbors(i, nx, ny, nz))
            visit(nb);
    });

// Irregular graph from adjacency list
det = num::connected_components(n_nodes,
    [&](int i) { return !excluded[i]; },
    [&](int i, auto&& visit) {
        for (int nb : adj[i]) visit(nb);
    });
```
