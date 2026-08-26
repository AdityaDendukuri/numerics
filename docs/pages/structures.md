# Discrete Structures {#page_structures}

The `structures` module provides union-find, addressable priority queues, degree queues, weighted graphs, canonical generators, and the traversal algorithms that run on them.

Graphs here carry no algebraic operations. Laplacians, adjacency matrices, and Markov generators are matrices, so they are assembled by `num::linear` (see @ref page_linear).

---

## 1. Disjoint-Set / Union-Find (num::DisjointSet)

`num::DisjointSet` maintains a collection of disjoint sets over elements \f$\{0, 1, \dots, N-1\}\f$ with near \f$\mathcal{O}(1)\f$ amortized cost per operation (\f$\mathcal{O}(\alpha(N))\f$ via two-pass path compression and union by rank, where \f$\alpha\f$ is the inverse Ackermann function).

### Basic Usage

```cpp
num::DisjointSet ds(6); // 6 singleton sets: {0}, {1}, {2}, {3}, {4}, {5}

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
std::vector<std::vector<num::idx>> partitions = ds.components();
```

### 32-Bit Index Specialization (num::DisjointSet32)

For large graphs (\f$N \ge 10^7\f$ nodes), `num::DisjointSet32` stores 32-bit integer indices, reducing memory consumption by 50% and doubling L3 cache line density:

```cpp
num::DisjointSet32 ds32(1000000); // Uses uint32_t internally (12 MB vs 24 MB)
```

---

## 2. Indexed Priority Queue (num::IndexedPriorityQueue)

`num::IndexedPriorityQueue` is an addressable binary heap where every item is identified by a unique index \f$i \in [0, \text{capacity})\f$. Unlike `std::priority_queue`, it supports \f$\mathcal{O}(\log N)\f$ priority updates (`improve_key` / `update`), \f$\mathcal{O}(1)\f$ presence queries (`contains`), and \f$\mathcal{O}(\log N)\f$ arbitrary element removal (`erase`).

### Min-Heap (Default)

```cpp
num::MinIndexedPQ<double> pq(100);

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
num::MaxIndexedPQ<double> max_pq(100);
max_pq.push(0, 0.4);
max_pq.push(1, 0.9);

num::idx best_state = max_pq.top_index(); // 1 (0.9 > 0.4)
```

---

## 3. Degree Queue (num::DegreeQueue)

Maintains integer degree buckets \f$d \in [0, d_{\max}]\f$ with doubly-linked lists for exact minimum-degree elimination in \f$\mathcal{O}(1)\f$ time:

```cpp
num::DegreeQueue dq(num_vertices, max_degree);
dq.insert(vertex_id, current_degree);

num::idx min_v = dq.pop_min(); // Extracts vertex with lowest degree in O(1)
dq.rekey(neighbor_id, old_deg, new_deg); // Degree update in O(1)
```

---

## 4. Graph Data Structure (num::Graph)

`num::Graph` stores directed or undirected graphs using weighted adjacency lists with fast degree and neighborhood accessors.

```cpp
num::Graph G(4, /*directed=*/false);

G.add_edge(0, 1, 2.5);
G.add_edge(1, 2, 1.0);
G.add_edge(2, 3, 4.0);

num::idx deg = G.degree(1);          // 2 incident edges
double w_deg = G.weighted_degree(1); // 3.5 (2.5 + 1.0)
```

### Compact Single-Precision Specialization (num::FloatGraph)

For large network graphs, point clouds, and mesh models, `num::FloatGraph` (`BasicGraph<float, uint32_t>`) cuts memory in half:

```cpp
num::FloatGraph mesh(100000); // 32-bit node indices and float edge weights
mesh.add_edge(0u, 1u, 1.5f);
```

---

## 5. Canonical Graph Generators (num::structures::*)

### Uniform Spanning Trees & Erdős–Rényi Networks
```cpp
std::mt19937_64 rng(42);

// Uniform Spanning Tree generated via loop-erased random walk (Wilson's algorithm)
num::Graph tree = num::structures::random_spanning_tree(100, rng, /*min_w=*/0.5, /*max_w=*/2.0);

// Erdős–Rényi random graph G(n, p) with guaranteed connectivity
num::Graph er = num::structures::erdos_renyi(100, 0.05, rng, /*ensure_connected=*/true);
```

### Structured Graph Families
```cpp
auto path  = num::structures::path_graph(10);
auto cycle = num::structures::cycle_graph(10);
auto grid  = num::structures::grid_2d(5, 5);       // 5x5 2D grid graph
auto star  = num::structures::star_graph(8);
auto kn    = num::structures::complete_graph(6);
```

---

## 6. Fundamental Graph Algorithms

```cpp
// 1. Fast connectivity check via Union-Find in O(E \alpha(V))
bool connected = num::structures::is_connected(G);

// 2. Connected components partition
std::vector<std::vector<num::idx>> components = num::structures::connected_components(G);

// 3. Single-Source Shortest Paths via Dijkstra + Indexed Priority Queue in O((V + E) log V)
std::vector<double> dist = num::structures::dijkstra(G, /*source=*/0);

// 4. Minimum Spanning Tree via Kruskal's algorithm in O(E log V)
num::Graph mst = num::structures::minimum_spanning_tree(G);

// 5. Breadth-first and depth-first search orderings
std::vector<num::idx> bfs_order = num::structures::bfs(G, 0);
std::vector<num::idx> dfs_order = num::structures::dfs(G, 0);
```

---

---

## 7. Compile-Time Concepts & Diagnostics

```cpp
static_assert(num::EquivalenceRelation<num::DisjointSet, num::idx>);
static_assert(num::AddressablePriorityQueue<num::MinIndexedPQ<double, num::idx>, double, num::idx>);
static_assert(num::IncidenceStructure<num::Graph, num::idx>);
```

The laws these structures maintain are sampled at runtime:

```cpp
num::DisjointSet ds(8);
ds.unite(0, 1);
ds.unite(1, 2);

num::structures::debug::verify_equivalence_relation(ds, num::idx(8)); // Reflexive, symmetric,
                                                                      // transitive, find canonical.

num::Graph G = num::structures::path_graph(10);
num::structures::debug::verify_degree_consistency(G); // degree(u) matches neighbors(u).
num::structures::debug::verify_handshake_lemma(G);    // sum of degrees == 2|E|.
```
