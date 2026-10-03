# Data structures

Graphs, queues, union-find and spatial neighbour structures. The algorithms on them are on the [Algorithms](algorithms.md) page.

## Graphs <structures/graph/graph.hpp>
num::graph

## Multigraphs <structures/graph/multigraph.hpp>
num::structures::multigraph num::structures::multi_edge num::structures::to_multigraph

## Grid graphs <structures/graph/structured_grid.hpp>
num::structured_grid_graph

## Union-find <structures/containers/disjoint_set.hpp>
num::disjoint_set num::basic_disjoint_set num::disjoint_set_32

## Indexed priority queues <structures/containers/indexed_priority_queue.hpp>
num::indexed_priority_queue num::min_indexed_pq num::max_indexed_pq

## Degree queues <structures/containers/degree_queue.hpp>
num::structures::degree_queue num::structures::basic_degree_queue num::structures::degree_queue_32

## Checks <structures/debug.hpp>
num::structures::debug::verify_handshake_lemma num::structures::debug::verify_laplacian_structure num::structures::debug::verify_degree_consistency num::structures::debug::verify_equivalence_relation num::structures::debug::verify_heap_order num::structures::debug::check_connected num::structures::debug::check_contains num::structures::debug::check_not_contains num::structures::debug::check_index_bounds num::structures::debug::check_vertex_bounds num::structures::debug::check_positive_weight

## 2D cell lists <spatial/cell_list.hpp>
num::cell_list_2d num::integer_range

## 3D cell lists <spatial/cell_list_3d.hpp>
num::cell_list_3d

## Verlet lists <spatial/verlet_list.hpp>
num::verlet_list_2d

## Periodic lattices <spatial/pbc_lattice.hpp>
num::pbc_lattice_2d

## SPH kernels <spatial/sph_kernel.hpp>
num::sph_kernel

## Spatial checks <spatial/debug.hpp>
num::spatial::debug::verify_kernel_normalization num::spatial::debug::verify_kernel_support num::spatial::debug::verify_lattice_symmetry
