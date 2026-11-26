/// @file structures/graph/multigraph.hpp
/// @brief Weighted multigraph with edge multiplicities.
#pragma once

#include "core/types.hpp"
#include "structures/graph/graph.hpp"
#include <algorithm>
#include <concepts>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace num::structures {

/// @brief Edge of a multigraph: target, weight and multiplicity.
template <typename Weight = double, std::integral Index = num::idx>
struct multi_edge {
    Index to{};
    Weight weight{1};
    std::uint8_t count = 1;
};

/// @brief Weighted graph that may have parallel edges, with counts.
template <typename Weight, std::integral Index>
class multigraph {
  public:
    using weight_type = Weight;
    using index_type = Index;
    using edge_type = multi_edge<Weight, Index>;

    /// @brief Construct a multigraph on `n` vertices with no edges.
    explicit multigraph(Index n = 0) : adj_(n) {}

    /// @brief Return the number of vertices \f$|V|\f$.
    [[nodiscard]] Index n_vertices() const noexcept {
        return static_cast<Index>(adj_.size());
    }

    /// @brief Add the undirected edge \f$\{u, v\}\f$.
    void add_edge(Index u, Index v, Weight weight = Weight{1}, std::uint8_t count = 1) {
        if (u >= adj_.size() || v >= adj_.size()) {
            throw std::out_of_range("multigraph::add_edge: vertex index out of range");
        }
        adj_[u].push_back({v, weight, count});
        adj_[v].push_back({u, weight, count});
    }

    /// @brief Add the directed edge \f$u \to v\f$.
    void add_directed_edge(Index u, Index v, Weight weight = Weight{1}, std::uint8_t count = 1) {
        if (u >= adj_.size() || v >= adj_.size()) {
            throw std::out_of_range("multigraph::add_directed_edge: vertex index out of range");
        }
        adj_[u].push_back({v, weight, count});
    }

    /// @brief Return the edges at `u`.
    [[nodiscard]] const array<edge_type> &neighbors(Index u) const {
        if (u >= adj_.size()) {
            throw std::out_of_range("multigraph::neighbors: vertex index out of range");
        }
        return adj_[u];
    }

    /// @brief Return the edges at `u`, for modification.
    [[nodiscard]] array<edge_type> &neighbors(Index u) {
        if (u >= adj_.size()) {
            throw std::out_of_range("multigraph::neighbors: vertex index out of range");
        }
        return adj_[u];
    }

    /// @brief Return the adjacency lists.
    [[nodiscard]] const array<array<edge_type>> &adjacency() const noexcept {
        return adj_;
    }

    /// @brief Return the adjacency lists, for modification.
    [[nodiscard]] array<array<edge_type>> &adjacency() noexcept {
        return adj_;
    }

    /// @brief Return the edges at `u`.
    [[nodiscard]] const array<edge_type> &operator[](Index u) const {
        return adj_[u];
    }
    [[nodiscard]] array<edge_type> &operator[](Index u) {
        return adj_[u];
    }

    /// @brief Merge parallel edges into a simple graph.
    [[nodiscard]] graph<Weight, Index> to_simple_graph() const {
        const Index n = n_vertices();
        graph<Weight, Index> G(n);
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
    array<array<edge_type>> adj_;
};

/// @brief Construct a multigraph from a graph.
template <typename Weight, std::integral Index>
[[nodiscard]] inline multigraph<Weight, Index>
to_multigraph(const graph<Weight, Index> &G) {
    const Index n = G.n_vertices();
    multigraph<Weight, Index> mg(n);
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
using structures::multi_edge;
using structures::multigraph;
} // namespace num
