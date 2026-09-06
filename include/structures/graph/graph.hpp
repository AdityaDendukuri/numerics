/// @file structures/graph/graph.hpp
/// @brief Weighted graph data structure with adjacency list and matrix/Laplacian conversions.
#pragma once

#include "core/types.hpp"
#include "structures/concepts.hpp"
#include "structures/debug.hpp"
#include <concepts>
#include <span>
#include <vector>

namespace num {

/// @brief Weighted directed or undirected graph represented as an adjacency list.
/// @tparam Weight scalar type of edge weights (e.g. double, float, int).
/// @tparam Index Integer type used for vertex indices (e.g. num::idx, uint32_t).
template <typename Weight = double, std::integral Index = num::idx>
class basic_graph {
  public:
    using weight_type = Weight;
    using index_type = Index;

    struct graph_edge {
        Index to{};
        Weight weight{1};
    };

    /// Construct a graph with n vertices.
    explicit basic_graph(Index n = 0, bool directed = false)
        : n_(n), m_(0), directed_(directed), adj_(n) {}

    /// Add an undirected edge between u and v with positive weight.
    void add_edge(Index u, Index v, Weight weight = Weight{1}) {
        structures::debug::check_vertex_bounds(u, n_, "graph::add_edge u");
        structures::debug::check_vertex_bounds(v, n_, "graph::add_edge v");
        structures::debug::check_positive_weight(weight, "graph::add_edge weight");

        adj_[u].push_back({v, weight});
        if (!directed_ && u != v) {
            adj_[v].push_back({u, weight});
        }
        m_++;
    }

    /// Add a directed edge from u to v with positive weight.
    void add_directed_edge(Index u, Index v, Weight weight = Weight{1}) {
        structures::debug::check_vertex_bounds(u, n_, "graph::add_directed_edge u");
        structures::debug::check_vertex_bounds(v, n_, "graph::add_directed_edge v");
        structures::debug::check_positive_weight(weight, "graph::add_directed_edge weight");

        adj_[u].push_back({v, weight});
        m_++;
    }

    /// Check whether an edge exists from u to v.
    [[nodiscard]] bool has_edge(Index u, Index v) const {
        structures::debug::check_vertex_bounds(u, n_, "graph::has_edge u");
        structures::debug::check_vertex_bounds(v, n_, "graph::has_edge v");
        for (const auto &e : adj_[u]) {
            if (e.to == v) return true;
        }
        return false;
    }

    /// Return the weight of edge (u, v), or 0 if no edge exists.
    [[nodiscard]] Weight edge_weight(Index u, Index v) const {
        structures::debug::check_vertex_bounds(u, n_, "graph::edge_weight u");
        structures::debug::check_vertex_bounds(v, n_, "graph::edge_weight v");
        for (const auto &e : adj_[u]) {
            if (e.to == v) return e.weight;
        }
        return Weight{0};
    }

    /// Return the number of incident edges for vertex u.
    [[nodiscard]] Index degree(Index u) const {
        structures::debug::check_vertex_bounds(u, n_, "graph::degree");
        return static_cast<Index>(adj_[u].size());
    }

    /// Return the sum of edge weights incident to vertex u.
    [[nodiscard]] Weight weighted_degree(Index u) const {
        structures::debug::check_vertex_bounds(u, n_, "graph::weighted_degree");
        Weight sum{0};
        for (const auto &e : adj_[u]) {
            sum += e.weight;
        }
        return sum;
    }

    /// Return view of neighbor edges for vertex u.
    [[nodiscard]] view<const graph_edge> neighbors(Index u) const {
        structures::debug::check_vertex_bounds(u, n_, "graph::neighbors");
        return adj_[u];
    }

    /// Return the total number of vertices.
    [[nodiscard]] Index n_vertices() const noexcept { return n_; }

    /// Return the total number of edges.
    [[nodiscard]] Index n_edges() const noexcept { return m_; }

    /// Return whether the graph is directed.
    [[nodiscard]] bool is_directed() const noexcept { return directed_; }







  private:
    Index n_ = 0;
    Index m_ = 0;
    bool directed_ = false;
    array<array<graph_edge>> adj_;
};

/// Default double-precision 64-bit graph alias
using graph = basic_graph<double, num::idx>;

/// Lightweight single-precision 32-bit graph alias (50% memory reduction)
using float_graph = basic_graph<float, uint32_t>;

static_assert(concepts::incidence_structure<graph, num::idx>, "graph must satisfy incidence_structure concept");
static_assert(concepts::incidence_structure<float_graph, uint32_t>, "float_graph must satisfy incidence_structure concept");

} // namespace num
