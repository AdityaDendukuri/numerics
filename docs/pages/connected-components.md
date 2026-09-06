# Connected Components {#page_connected_components}

`include/spatial/connected_components.hpp` provides `num::connected_components`,
a template BFS labelling function with a pre-allocated flat queue
(no heap allocation per call, no recursion).

---

## Problem

Connected-component labeling partitions included vertices of a finite graph into
maximal connected subsets.  For Ising nucleation, included vertices are the
spin-down sites and the largest component is the nucleus observable.

`connected_components` uses caller-supplied predicates for site inclusion and
neighbor traversal.

---

## Routine Reference

```cpp
struct num::cluster_result {
    num::array<int> id;        // id[i]: -2 = excluded, >=0 = cluster index
    num::array<int> sizes;     // sizes[c] = number of sites in cluster c
    int largest_id   = -1;
    int largest_size = 0;
};

template<typename InCluster, typename Neighbors>
num::cluster_result num::connected_components(
    int n_sites,
    InCluster&&  in_cluster,   // bool(int i)        -- include site i?
    Neighbors&&  neighbors);   // void(int i, F&& f) -- call f(nb) per neighbor
```

### Labels

| `id[i]` | Meaning |
|---|---|
| `-2` | Excluded -- `in_cluster(i)` returned `false` |
| `>=0` | Cluster index |

Sites with `id[i] == cluster_result::largest_id` belong to the largest cluster.

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

No heap allocations after the initial `cluster_result` construction.

---

## Ising Nucleation Observable

```cpp
num::pbc_lattice_2d nbr(N);
num::cluster_result det;

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

## Generalizations

The callable interface accepts any graph topology:

```cpp
// 3D cubic lattice with PBC
const int nx = 8, ny = 8, nz = 8;
std::vector<char> active(nx * ny * nz, 1);
auto six_neighbors = [](int i, int, int, int) { return std::vector<int>{i}; };

auto det = num::connected_components(nx*ny*nz,
    [&](int i) { return active[i]; },
    [&](int i, auto&& visit) {
        for (int nb : six_neighbors(i, nx, ny, nz))
            visit(nb);
    });

// Irregular graph from adjacency list
const int n_nodes = 16;
std::vector<char> excluded(n_nodes, 0);
std::vector<std::vector<int>> adjacency(n_nodes);

auto det2 = num::connected_components(n_nodes,
    [&](int i) { return !excluded[i]; },
    [&](int i, auto&& visit) {
        for (int nb : adjacency[i]) visit(nb);
    });
```

---

## See Also

* **`num::structures::connected_components` (@ref page_structures):** graph topology partitioning on `num::graph` using Union-Find.
* **`num::disjoint_set` (@ref page_structures):** General \f$\mathcal{O}(\alpha(N))\f$ disjoint set data structure with path compression.
