/// @file structures/graph/graph.hpp
/// @brief Weighted graph as an adjacency list.
#pragma once

#include "core/types.hpp"
#include "structures/concepts.hpp"
#include "structures/debug.hpp"
#include <concepts>
#include <span>
#include <vector>

namespace num {

/// @brief Weighted graph \f$G = (V, E, w)\f$, directed or undirected.
template <typename Weight, std::integral Index>
class graph {
  public:
    using weight_type = Weight;
    using index_type = Index;

    struct graph_edge {
        Index to{};
        Weight weight{1};
    };

    /// @brief Graph on `n` vertices with no edges.
    explicit graph(Index n = 0, bool directed = false)
        : n_(n), m_(0), directed_(directed), adj_(n) {}

    /// @brief Add undirected edge \f$\{u, v\}\f$ with weight \f$w > 0\f$.
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

    /// @brief Add directed edge \f$u \to v\f$ with weight \f$w > 0\f$.
    void add_directed_edge(Index u, Index v, Weight weight = Weight{1}) {
        structures::debug::check_vertex_bounds(u, n_, "graph::add_directed_edge u");
        structures::debug::check_vertex_bounds(v, n_, "graph::add_directed_edge v");
        structures::debug::check_positive_weight(weight, "graph::add_directed_edge weight");

        adj_[u].push_back({v, weight});
        m_++;
    }

    /// @brief Whether edge \f$u \to v\f$ exists.
    [[nodiscard]] bool has_edge(Index u, Index v) const {
        structures::debug::check_vertex_bounds(u, n_, "graph::has_edge u");
        structures::debug::check_vertex_bounds(v, n_, "graph::has_edge v");
        for (const auto &e : adj_[u]) {
            if (e.to == v) return true;
        }
        return false;
    }

    /// @brief Weight of \f$u \to v\f$, or 0 without the edge.
    [[nodiscard]] Weight edge_weight(Index u, Index v) const {
        structures::debug::check_vertex_bounds(u, n_, "graph::edge_weight u");
        structures::debug::check_vertex_bounds(v, n_, "graph::edge_weight v");
        for (const auto &e : adj_[u]) {
            if (e.to == v) return e.weight;
        }
        return Weight{0};
    }

    /// @brief Degree: \f$\deg(u)\f$.
    [[nodiscard]] Index degree(Index u) const {
        structures::debug::check_vertex_bounds(u, n_, "graph::degree");
        return static_cast<Index>(adj_[u].size());
    }

    /// @brief Weighted degree: \f$\sum_v w_{uv}\f$.
    [[nodiscard]] Weight weighted_degree(Index u) const {
        structures::debug::check_vertex_bounds(u, n_, "graph::weighted_degree");
        Weight sum{0};
        for (const auto &e : adj_[u]) {
            sum += e.weight;
        }
        return sum;
    }

    /// @brief Edges incident to `u`.
    [[nodiscard]] view<const graph_edge> neighbors(Index u) const {
        structures::debug::check_vertex_bounds(u, n_, "graph::neighbors");
        return adj_[u];
    }

    /// @brief Vertex count \f$|V|\f$.
    [[nodiscard]] Index n_vertices() const noexcept { return n_; }

    /// @brief Edge count \f$|E|\f$.
    [[nodiscard]] Index n_edges() const noexcept { return m_; }

    /// @brief Whether the graph is directed.
    [[nodiscard]] bool is_directed() const noexcept { return directed_; }

  private:
    Index n_ = 0;
    Index m_ = 0;
    bool directed_ = false;
    array<array<graph_edge>> adj_;
};

static_assert(concepts::incidence_structure<graph<double, num::idx>, num::idx>,
              "graph must satisfy incidence_structure concept");
static_assert(concepts::incidence_structure<graph<float, uint32_t>, uint32_t>,
              "graph must satisfy incidence_structure concept");

static_assert(incidence_structure<graph<real, idx>>);
static_assert(weighted_incidence<graph<real, idx>::graph_edge>);
} // namespace num
