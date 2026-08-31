/// @file structures/algorithms/traversal.hpp
/// @brief Fundamental graph algorithms: connectivity, Dijkstra, Kruskal MST, BFS, and DFS.
#pragma once

#include "core/types.hpp"
#include "structures/containers/disjoint_set.hpp"
#include "structures/containers/indexed_priority_queue.hpp"
#include "structures/debug.hpp"
#include "structures/graph/graph.hpp"
#include <algorithm>
#include <concepts>
#include <limits>
#include <queue>
#include <type_traits>
#include <vector>

namespace num::structures {

/// Check if graph is connected.
template <typename Weight, std::integral Index>
[[nodiscard]] inline bool is_connected(const BasicGraph<Weight, Index> &G) {
    const Index n = G.n_vertices();
    if (n <= 1) return true;

    BasicDisjointSet<Index> ds(n);
    for (Index u = 0; u < n; ++u) {
        for (const auto &e : G.neighbors(u)) {
            ds.unite(u, e.to);
        }
    }
    return ds.count() == 1;
}

/// Compute connected components partition.
template <typename Weight, std::integral Index>
[[nodiscard]] inline std::vector<std::vector<Index>> connected_components(const BasicGraph<Weight, Index> &G) {
    const Index n = G.n_vertices();
    BasicDisjointSet<Index> ds(n);
    for (Index u = 0; u < n; ++u) {
        for (const auto &e : G.neighbors(u)) {
            ds.unite(u, e.to);
        }
    }
    return ds.components();
}

/// Compute single-source shortest path distances using Dijkstra with an Indexed Priority Queue.
template <typename Weight, std::integral Index, std::integral Source = Index>
[[nodiscard]] inline std::vector<Weight> dijkstra(const BasicGraph<Weight, Index> &G, Source source_in) {
    const Index n = G.n_vertices();
    const Index source = static_cast<Index>(source_in);
    debug::check_vertex_bounds(source, n, "structures::dijkstra source");

    constexpr Weight inf = std::numeric_limits<Weight>::infinity();
    std::vector<Weight> dist(n, inf);
    dist[source] = Weight{0};

    MinIndexedPQ<Weight, Index> pq(n);
    pq.push(source, Weight{0});

    while (!pq.empty()) {
        Index u = pq.top_index();
        Weight d = pq.top_key();
        pq.pop();

        for (const auto &e : G.neighbors(u)) {
            debug::check_positive_weight(e.weight, "structures::dijkstra edge weight");
            Weight new_dist = d + e.weight;
            if (new_dist < dist[e.to]) {
                dist[e.to] = new_dist;
                pq.update(e.to, new_dist);
            }
        }
    }
    return dist;
}

/// Breadth-first search traversal order starting from source.
template <typename Weight, std::integral Index, std::integral Source = Index>
[[nodiscard]] inline std::vector<Index> bfs(const BasicGraph<Weight, Index> &G, Source source_in) {
    const Index n = G.n_vertices();
    const Index source = static_cast<Index>(source_in);
    debug::check_vertex_bounds(source, n, "structures::bfs source");

    std::vector<bool> visited(n, false);
    std::vector<Index> order;
    order.reserve(n);

    std::queue<Index> q;
    q.push(source);
    visited[source] = true;

    while (!q.empty()) {
        Index u = q.front();
        q.pop();
        order.push_back(u);

        for (const auto &e : G.neighbors(u)) {
            if (!visited[e.to]) {
                visited[e.to] = true;
                q.push(e.to);
            }
        }
    }
    return order;
}

/// Depth-first search traversal order starting from source.
template <typename Weight, std::integral Index, std::integral Source = Index>
[[nodiscard]] inline std::vector<Index> dfs(const BasicGraph<Weight, Index> &G, Source source_in) {
    const Index n = G.n_vertices();
    const Index source = static_cast<Index>(source_in);
    debug::check_vertex_bounds(source, n, "structures::dfs source");

    std::vector<bool> visited(n, false);
    std::vector<Index> order;
    order.reserve(n);

    std::vector<Index> stack = {source};

    while (!stack.empty()) {
        Index u = stack.back();
        stack.pop_back();

        if (visited[u]) continue;
        visited[u] = true;
        order.push_back(u);

        for (const auto &e : G.neighbors(u)) {
            if (!visited[e.to]) {
                stack.push_back(e.to);
            }
        }
    }
    return order;
}

/// Compute Minimum Spanning Tree (MST) using Kruskal's algorithm with DisjointSet.
template <typename Weight, std::integral Index>
[[nodiscard]] inline BasicGraph<Weight, Index> minimum_spanning_tree(const BasicGraph<Weight, Index> &G) {
    const Index n = G.n_vertices();
    struct KruskalEdge {
        Index u, v;
        Weight weight;
        bool operator<(const KruskalEdge &o) const { return weight < o.weight; }
    };

    std::vector<KruskalEdge> edges;
    for (Index u = 0; u < n; ++u) {
        for (const auto &e : G.neighbors(u)) {
            if (u < e.to || G.is_directed()) {
                edges.push_back({u, e.to, e.weight});
            }
        }
    }
    std::sort(edges.begin(), edges.end());

    BasicDisjointSet<Index> ds(n);
    BasicGraph<Weight, Index> mst(n, G.is_directed());

    for (const auto &e : edges) {
        if (ds.unite(e.u, e.v)) {
            mst.add_edge(e.u, e.v, e.weight);
            if (mst.n_edges() == n - 1) {
                break;
            }
        }
    }
    return mst;
}

} // namespace num::structures
