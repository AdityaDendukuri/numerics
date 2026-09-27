# Discrete Structures {#page_structures}

The `structures` module provides union-find, addressable priority queues, degree queues, weighted graphs, canonical generators, and the traversal algorithms that run on them.

Graphs here carry no algebraic operations. Laplacians, adjacency matrices, and Markov generators are matrices, so they are assembled by `num::linear` (see @ref page_linear).

## Container vocabulary

The graph and traversal routines on this page return `num::array` where they used to
return `std::vector` -- the same type under a name that does not collide with
`num::vec`. The full vocabulary, scalars included, is documented on the containers
page: see @ref page_container.

---

## 1. Fixed-Capacity Multi-Index (num::multi_index)

`num::multi_index` is a short integer tuple that names a point in a discrete state
space — an occupancy vector for a chemical master equation, a lattice site, a
multi-dimensional array subscript. It holds up to `num::multi_index::k_max_dim` (8)
`int` coordinates inline, so it is 36 bytes, never allocates, and copies by value.

```cpp
num::multi_index s{2, 0, 1};      // a 3-dimensional state
num::idx d = s.size();            // 3 active coordinates
int first = s[0];                 // 2

num::multi_index step{1, 0, 0};
num::multi_index next = s + step; // {3, 0, 1}; coordinates add elementwise
```

Coordinate counts are not checked against each other in `operator+` and `operator-`;
both use the left operand's size. Constructing past `k_max_dim` trips an `assert`, so
a state space wider than eight dimensions needs a different representation.

It is both hashable and ordered, which is what makes it usable as a key:

```cpp
// sparse state spaces are enumerated, not indexed, so states are stored as keys
num::unordered_map<num::multi_index, double> probability; // hashed, O(1) average
probability[{2, 0, 1}] = 0.25;

num::map<num::multi_index, double> ordered;                // key order, for reproducible sweeps
ordered[{2, 0, 1}] = 0.25;
```

`operator<` orders by coordinate count first and then lexicographically, so indices of
different dimension never compare equal and a `num::map` keyed on them iterates in a
stable, reproducible order.

---

## 2. Disjoint-Set / Union-Find (num::disjoint_set)

`num::disjoint_set` maintains a collection of disjoint sets over elements \f$\{0, 1, \dots, N-1\}\f$ with near \f$\mathcal{O}(1)\f$ amortized cost per operation (\f$\mathcal{O}(\alpha(N))\f$ via two-pass path compression and union by rank, where \f$\alpha\f$ is the inverse Ackermann function).

### Basic Usage

```cpp
num::disjoint_set ds(6); // 6 singleton sets: {0}, {1}, {2}, {3}, {4}, {5}

ds.unite(0, 1);
ds.unite(1, 2);
ds.unite(3, 4);

bool c1 = ds.connected(0, 2); // true
bool c2 = ds.connected(0, 3); // false

num::idx comps = ds.count();           // 3 disjoint components
num::idx sz = ds.component_size(0);     // 3 elements in component {0, 1, 2}
```

### Component Extraction

```cpp
// Extract all connected components as lists of indices
num::array<num::array<num::idx>> partitions = ds.components();
```

### 32-Bit Index Specialization (num::disjoint_set_32)

For large graphs (\f$N \ge 10^7\f$ nodes), `num::disjoint_set_32` stores 32-bit integer indices, reducing memory consumption by 50% and doubling L3 cache line density:

```cpp
num::disjoint_set_32 ds32(1000000); // Uses uint32_t internally (12 MB vs 24 MB)
```

---

## 3. Indexed Priority Queue (num::indexed_priority_queue)

`num::indexed_priority_queue` is an addressable binary heap where every item is identified by a unique index \f$i \in [0, \text{capacity})\f$. Unlike `std::priority_queue`, it supports \f$\mathcal{O}(\log N)\f$ priority updates (`improve_key` / `update`), \f$\mathcal{O}(1)\f$ presence queries (`contains`), and \f$\mathcal{O}(\log N)\f$ arbitrary element removal (`erase`).

### Min-Heap (Default)

```cpp
num::min_indexed_pq<double> pq(100);

pq.push(0, 10.5);
pq.push(1, 3.2);
pq.push(2, 7.8);

num::idx min_idx = pq.top_index(); // 1
double min_val = pq.top_key();     // 3.2

// Improve key (e.g. Dijkstra edge relaxation)
pq.improve_key(2, 1.5);
num::idx new_top = pq.top_index(); // 2 (since 1.5 < 3.2)

pq.pop(); // Removes element 2
```

### Max-Heap

```cpp
// For largest-weight first frontier exploration
num::max_indexed_pq<double> max_pq(100);
max_pq.push(0, 0.4);
max_pq.push(1, 0.9);

num::idx best_state = max_pq.top_index(); // 1 (0.9 > 0.4)
```

---

## 4. Degree Queue (num::degree_queue)

Maintains integer degree buckets \f$d \in [0, d_{\max}]\f$ with doubly-linked lists for exact minimum-degree elimination in \f$\mathcal{O}(1)\f$ time:

```cpp
num::degree_queue dq(num_vertices, max_degree);
dq.insert(vertex_id, current_degree);

num::idx min_v = dq.pop_min(); // Extracts vertex with lowest degree in O(1)
dq.rekey(neighbor_id, old_deg, new_deg); // Degree update in O(1)
```

---

## 5. graph Data Structure (num::graph)

`num::graph` stores directed or undirected graphs using weighted adjacency lists with fast degree and neighborhood accessors.

```cpp
num::graph G(4, /*directed=*/false);

G.add_edge(0, 1, 2.5);
G.add_edge(1, 2, 1.0);
G.add_edge(2, 3, 4.0);

num::idx deg = G.degree(1);          // 2 incident edges
double w_deg = G.weighted_degree(1); // 3.5 (2.5 + 1.0)
```

### Compact Single-Precision Specialization (num::float_graph)

For large network graphs, point clouds, and mesh models, `num::float_graph` (`basic_graph<float, uint32_t>`) cuts memory in half:

```cpp
num::float_graph mesh(100000); // 32-bit node indices and float edge weights
mesh.add_edge(0u, 1u, 1.5f);
```

---

## 6. Canonical graph Generators (num::structures::*)

### Uniform Spanning Trees & Erdős–Rényi Networks
```cpp
num::rng64 rng(42);

// Uniform Spanning Tree generated via loop-erased random walk (Wilson's algorithm)
num::graph tree = num::structures::random_spanning_tree(100, rng, /*min_w=*/0.5, /*max_w=*/2.0);

// Erdős–Rényi random graph G(n, p) with guaranteed connectivity
num::graph er = num::structures::erdos_renyi(100, 0.05, rng, /*ensure_connected=*/true);
```

### Structured graph Families
```cpp
auto path  = num::structures::path_graph(10);
auto cycle = num::structures::cycle_graph(10);
auto grid  = num::structures::grid_2d(5, 5);       // 5x5 2D grid graph
auto star  = num::structures::star_graph(8);
auto kn    = num::structures::complete_graph(6);
```

---

## 7. Fundamental graph Algorithms

```cpp
// 1. Fast connectivity check via Union-Find in O(E \alpha(V))
bool connected = num::structures::is_connected(G);

// 2. Connected components partition
num::array<num::array<num::idx>> components = num::structures::connected_components(G);

// 3. Single-Source Shortest Paths via Dijkstra + Indexed Priority Queue in O((V + E) log V)
num::array<double> dist = num::structures::dijkstra(G, /*source=*/0);

// 4. Minimum Spanning Tree via Kruskal's algorithm in O(E log V)
num::graph mst = num::structures::minimum_spanning_tree(G);

// 5. Breadth-first and depth-first search orderings
num::array<num::idx> bfs_order = num::structures::bfs(G, 0);
num::array<num::idx> dfs_order = num::structures::dfs(G, 0);
```

---

## 8. Concepts & Diagnostics

```cpp
static_assert(num::equivalence_relation<num::disjoint_set, num::idx>);
static_assert(num::addressable_priority_queue<num::min_indexed_pq<double, num::idx>, double, num::idx>);
static_assert(num::incidence_structure<num::graph, num::idx>);
```

Runtime law verification:
```cpp
num::disjoint_set ds(8);
ds.unite(0, 1);
ds.unite(1, 2);
num::structures::debug::verify_equivalence_relation(ds, num::idx(8));

num::graph G = num::structures::path_graph(10);
num::structures::debug::verify_degree_consistency(G);
num::structures::debug::verify_handshake_lemma(G);
```


## 9. Spanning-tree clique sampling

`structures/graph/clique.hpp` replaces the fill of one elimination by a random spanning tree
of it, reweighted so the elimination stays unbiased in expectation.

### The symmetric clique

Eliminating `v` adds a clique on its neighbours with Schur-complement weights
\f$w_{ij} = c_i c_j / C\f$, where \f$c_i\f$ is the conductance from `v` to neighbour `i`
and \f$C = \sum_i c_i\f$. For a weighted uniform spanning tree, Kirchhoff gives the inclusion
probability \f$p_e = w_e R_{\mathrm{eff}}(e)\f$. Effective resistances usually need
Laplacian solves, but this clique Laplacian is \f$\operatorname{diag}(c) - cc^{T}/C\f$, and
solving it gives the series path through `v`:
\f[
  R_{\mathrm{eff}}(i,j) = \frac{1}{c_i} + \frac{1}{c_j}, \qquad
  p_{ij} = \frac{c_i + c_j}{C}, \qquad
  \sum_{i<j} p_{ij} = d - 1.
\f]
The sum is the edge count of a spanning tree. The reweighting
\f$\widehat c_{ij} = w_{ij}/p_{ij} = c_i c_j / (c_i + c_j)\f$ is the harmonic mean, the same
series conductance the independent sampler assigns, so
\f$\mathbb{E}[\widetilde L^{(v)}] = \mathrm{Sc}(L)\f$.

Aldous--Broder degenerates on this clique. Its transition kernel
\f$P(i \to j) = c_j / (C - c_i)\f$ does not depend on the current vertex, so the walk is a
coupon collector: draw i.i.d. from \f$c/C\f$ and record the entering edge at each first visit.

### The directed biclique

A nonsymmetric pivot leaves the rank-one update \f$xy^{T}/a\f$, with \f$x_i = -A_{iv}\f$,
\f$y_j = -A_{vj}\f$ and \f$a = A_{vv}\f$. Lifting to a bipartite graph with vertices
\f$i_L\f$ for in-neighbours, \f$j_R\f$ for out-neighbours, and conductances \f$x_iy_j/a\f$
keeps the two directions of an edge distinct. With \f$X = \sum_i x_i\f$ and
\f$Y = \sum_j y_j\f$,
\f[
  R_{\mathrm{eff}}(i_L, j_R) = a\Big[\tfrac{1}{x_iY} + \tfrac{1}{y_jX} - \tfrac{1}{XY}\Big],
  \qquad
  p_{ij} = 1 - (1-\alpha_i)(1-\beta_j),
\f]
where \f$\alpha_i = x_i/X\f$ and \f$\beta_j = y_j/Y\f$. The probabilities sum to \f$m + n - 1\f$,
the edge count of a spanning tree on the duplicated vertices. The walk alternates the fixed
distributions \f$y/Y\f$ and \f$x/X\f$. The conductance reading needs \f$x_i, y_j, a > 0\f$.
Matrix Chernoff bounds for strongly Rayleigh measures are Hermitian results, so they say
nothing about concentration for this nonsymmetric update.

### Measured behaviour

On 2D grid Laplacians with n up to 40k, one tree per elimination is worse than the
independent sampler: 44 PCG iterations against 41 for `ac1` and 29 for `ac2`. On
higher-degree graphs with skewed weights the coupon collector also makes setup several
times more expensive.

Averaging k trees per elimination concentrates, as the strongly Rayleigh Chernoff theory
predicts. On a 14400-vertex grid it takes 44, 27, 20 and 15 iterations at k = 1, 2, 4 and 8,
against 29 for `ac2`. Setup grows faster than k, because each tree raises the degree of later
eliminations: 11 ms, 82 ms, 616 ms and 2.7 s. It pays only when one factorization serves many
solves, about 50 to break even at k = 2 and 600 at k = 8.
