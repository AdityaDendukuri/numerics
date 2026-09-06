/// @file structures/graph/multigraph.hpp
/// @brief Weighted multigraph data structure with multi-edge multiplicities for RandNLA and Laplacian solvers.
#pragma once

#include "container/matrix.hpp"
#include "core/types.hpp"
#include "structures/graph/graph.hpp"
#include <algorithm>
#include <concepts>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace num::structures {

/// Weighted multi-edge with multiplicity count.
template <typename Weight = double, std::integral Index = num::idx>
struct multi_edge {
    Index to{};
    Weight weight{1};
    std::uint8_t count = 1;
};

/// @brief Weighted multigraph representation supporting parallel edges and integer counts.
template <typename Weight = double, std::integral Index = num::idx>
class basic_multigraph {
  public:
    using weight_type = Weight;
    using index_type = Index;
    using edge_type = multi_edge<Weight, Index>;

    /// Construct an empty multigraph with n vertices.
    explicit basic_multigraph(Index n = 0) : adj_(n) {}

    /// Number of vertices.
    [[nodiscard]] Index n_vertices() const noexcept {
        return static_cast<Index>(adj_.size());
    }

    /// Add an undirected multi-edge between u and v.
    void add_edge(Index u, Index v, Weight weight = Weight{1}, std::uint8_t count = 1) {
        if (u >= adj_.size() || v >= adj_.size()) {
            throw std::out_of_range("basic_multigraph::add_edge: vertex index out of range");
        }
        adj_[u].push_back({v, weight, count});
        adj_[v].push_back({u, weight, count});
    }

    /// Add a directed multi-edge from u to v.
    void add_directed_edge(Index u, Index v, Weight weight = Weight{1}, std::uint8_t count = 1) {
        if (u >= adj_.size() || v >= adj_.size()) {
            throw std::out_of_range("basic_multigraph::add_directed_edge: vertex index out of range");
        }
        adj_[u].push_back({v, weight, count});
    }

    /// Const access to neighbor list of vertex u.
    [[nodiscard]] const std::vector<edge_type> &neighbors(Index u) const {
        if (u >= adj_.size()) {
            throw std::out_of_range("basic_multigraph::neighbors: vertex index out of range");
        }
        return adj_[u];
    }

    /// Mutable access to neighbor list of vertex u.
    [[nodiscard]] std::vector<edge_type> &neighbors(Index u) {
        if (u >= adj_.size()) {
            throw std::out_of_range("basic_multigraph::neighbors: vertex index out of range");
        }
        return adj_[u];
    }

    /// Raw underlying adjacency list.
    [[nodiscard]] const std::vector<std::vector<edge_type>> &adjacency() const noexcept {
        return adj_;
    }

    /// Mutable raw underlying adjacency list.
    [[nodiscard]] std::vector<std::vector<edge_type>> &adjacency() noexcept {
        return adj_;
    }

    /// Subscript operator for vertex neighbors.
    [[nodiscard]] const std::vector<edge_type> &operator[](Index u) const {
        return adj_[u];
    }
    [[nodiscard]] std::vector<edge_type> &operator[](Index u) {
        return adj_[u];
    }

    /// Convert to simple basic_graph (consolidating parallel edges).
    [[nodiscard]] basic_graph<Weight, Index> to_simple_graph() const {
        const Index n = n_vertices();
        basic_graph<Weight, Index> G(n);
        for (Index u = 0; u < n; ++u) {
            for (const auto &e : adj_[u]) {
                if (u < e.to) {
                    G.add_edge(u, e.to, e.weight);
                }
            }
        }
        return G;
    }


  private:
    std::vector<std::vector<edge_type>> adj_;
};

using multigraph = basic_multigraph<double, num::idx>;
using float_multigraph = basic_multigraph<float, std::uint32_t>;

/// Convert a basic_graph to a basic_multigraph.
template <typename Weight, std::integral Index>
[[nodiscard]] inline basic_multigraph<Weight, Index>
to_multigraph(const basic_graph<Weight, Index> &G) {
    const Index n = G.n_vertices();
    basic_multigraph<Weight, Index> mg(n);
    for (Index u = 0; u < n; ++u) {
        for (const auto &e : G.neighbors(u)) {
            if (u < e.to) {
                mg.add_edge(u, e.to, e.weight, 1);
            }
        }
    }
    return mg;
}



} // namespace num::structures

namespace num {
using structures::basic_multigraph;
using structures::float_multigraph;
using structures::multi_edge;
using structures::multigraph;
} // namespace num
