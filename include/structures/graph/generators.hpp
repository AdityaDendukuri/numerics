/// @file structures/graph/generators.hpp
/// @brief Canonical random and structured graph generators.
#pragma once

#include "core/types.hpp"
#include "structures/containers/disjoint_set.hpp"
#include "structures/debug.hpp"
#include "structures/graph/graph.hpp"
#include <algorithm>
#include <concepts>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

namespace num::structures {

/// Generate an undirected path graph: 0 - 1 - 2 - ... - (n-1).
template <typename Weight = double, std::integral Index = num::idx>
[[nodiscard]] inline basic_graph<Weight, Index> path_graph(std::type_identity_t<Index> n,
                                                         Weight weight = Weight{1}) {
    basic_graph<Weight, Index> G(n);
    for (Index i = 0; i + 1 < n; ++i) {
        G.add_edge(i, i + 1, weight);
    }
    return G;
}

/// Generate an undirected cycle graph: 0 - 1 - ... - (n-1) - 0.
template <typename Weight = double, std::integral Index = num::idx>
[[nodiscard]] inline basic_graph<Weight, Index> cycle_graph(std::type_identity_t<Index> n,
                                                          Weight weight = Weight{1}) {
    basic_graph<Weight, Index> G(n);
    for (Index i = 0; i < n; ++i) {
        G.add_edge(i, (i + 1) % n, weight);
    }
    return G;
}

/// Generate an undirected 2D grid graph of size nx by ny.
template <typename Weight = double, std::integral Index = num::idx>
[[nodiscard]] inline basic_graph<Weight, Index> grid_2d(std::type_identity_t<Index> nx,
                                                      std::type_identity_t<Index> ny,
                                                      Weight weight = Weight{1}) {
    const Index n = nx * ny;
    basic_graph<Weight, Index> G(n);
    const auto id = [&](Index x, Index y) { return (y * nx) + x; };

    for (Index y = 0; y < ny; ++y) {
        for (Index x = 0; x < nx; ++x) {
            if (x + 1 < nx) G.add_edge(id(x, y), id(x + 1, y), weight);
            if (y + 1 < ny) G.add_edge(id(x, y), id(x, y + 1), weight);
        }
    }
    return G;
}

/// Generate an undirected complete graph K_n.
template <typename Weight = double, std::integral Index = num::idx>
[[nodiscard]] inline basic_graph<Weight, Index> complete_graph(std::type_identity_t<Index> n,
                                                             Weight weight = Weight{1}) {
    basic_graph<Weight, Index> G(n);
    for (Index i = 0; i < n; ++i) {
        for (Index j = i + 1; j < n; ++j) {
            G.add_edge(i, j, weight);
        }
    }
    return G;
}

/// Generate a star graph with center at vertex 0 and n-1 leaves.
template <typename Weight = double, std::integral Index = num::idx>
[[nodiscard]] inline basic_graph<Weight, Index> star_graph(std::type_identity_t<Index> n,
                                                         Weight weight = Weight{1}) {
    basic_graph<Weight, Index> G(n);
    for (Index i = 1; i < n; ++i) {
        G.add_edge(0, i, weight);
    }
    return G;
}

/// Generate a uniformly random connected spanning tree on n vertices via Wilson's algorithm.
template <typename Weight = double, std::integral Index = num::idx, typename RNG>
[[nodiscard]] inline basic_graph<Weight, Index> random_spanning_tree(std::type_identity_t<Index> n,
                                                                   RNG &rng,
                                                                   Weight min_weight = Weight{1},
                                                                   Weight max_weight = Weight{1}) {
    basic_graph<Weight, Index> G(n);
    if (n <= 1) return G;

    std::uniform_real_distribution<double> weight_dist(static_cast<double>(min_weight),
                                                       static_cast<double>(max_weight));

    // Wilson's algorithm / Loop-Erased Random Walk for uniform spanning tree in O(n)
    array<bool> in_tree(n, false);
    array<Index> next(n);

    // Root the tree at vertex 0
    in_tree[0] = true;
    std::uniform_int_distribution<Index> vertex_dist(0, n - 1);

    for (Index i = 1; i < n; ++i) {
        Index u = i;
        while (!in_tree[u]) {
            Index v = vertex_dist(rng);
            while (v == u) {
                v = vertex_dist(rng);
            }
            next[u] = v;
            u = v;
        }

        // Add loop-erased path to tree
        u = i;
        while (!in_tree[u]) {
            in_tree[u] = true;
            Index v = next[u];
            Weight w = (min_weight == max_weight)
                           ? min_weight
                           : static_cast<Weight>(weight_dist(rng));
            G.add_edge(u, v, w);
            u = v;
        }
    }
    return G;
}

/// Generate an Erdős-Rényi random graph G(n, p) with optional guaranteed connectivity.
template <typename Weight = double, std::integral Index = num::idx, typename RNG>
[[nodiscard]] inline basic_graph<Weight, Index> erdos_renyi(std::type_identity_t<Index> n,
                                                           double p, RNG &rng,
                                                           bool ensure_connected = true,
                                                           Weight min_weight = Weight{1},
                                                           Weight max_weight = Weight{1}) {
    std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
    std::uniform_real_distribution<double> weight_dist(static_cast<double>(min_weight),
                                                       static_cast<double>(max_weight));

    basic_graph<Weight, Index> G = ensure_connected
                                      ? random_spanning_tree<Weight, Index>(n, rng, min_weight, max_weight)
                                      : basic_graph<Weight, Index>(n);

    for (Index i = 0; i < n; ++i) {
        for (Index j = i + 1; j < n; ++j) {
            if (!G.has_edge(i, j) && prob_dist(rng) < p) {
                Weight w = (min_weight == max_weight)
                               ? min_weight
                               : static_cast<Weight>(weight_dist(rng));
                G.add_edge(i, j, w);
            }
        }
    }
    return G;
}

} // namespace num::structures
