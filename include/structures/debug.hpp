/// @file structures/debug.hpp
/// @brief Runtime verification of the laws discrete structures maintain.
#pragma once

#include "core/debug.hpp"
#include "core/types.hpp"
#include <cmath>
#include <concepts>
#include <source_location>
#include <string>
#include <string_view>

namespace num::structures::debug {

/// @brief Verify index is within valid capacity bounds [0, capacity).
template <std::integral Index>
inline void check_index_bounds(Index index, Index capacity, std::string_view label = "Index",
                               std::source_location loc = std::source_location::current()) {
    if (num::debug::get_level() == num::debug::diagnostic_level::off) {
        return;
    }
    if (index >= capacity) {
        std::string msg = std::string(label) + " out of bounds: index " + std::to_string(index) +
                          " >= capacity " + std::to_string(capacity);
        num::debug::panic("IndexOutOfBoundsError", msg, loc);
    }
}

/// @brief Verify an indexed item exists in an indexed container.
template <std::integral Index>
inline void check_contains(bool exists, Index index,
                           std::string_view label = "indexed_priority_queue",
                           std::source_location loc = std::source_location::current()) {
    if (num::debug::get_level() == num::debug::diagnostic_level::off) {
        return;
    }
    if (!exists) {
        std::string msg = std::string(label) + ": element with index " + std::to_string(index) +
                          " does not exist";
        num::debug::panic("KeyNotFoundError", msg, loc);
    }
}

/// @brief Verify an indexed item does NOT already exist before insertion.
template <std::integral Index>
inline void check_not_contains(bool exists, Index index,
                               std::string_view label = "indexed_priority_queue",
                               std::source_location loc = std::source_location::current()) {
    if (num::debug::get_level() == num::debug::diagnostic_level::off) {
        return;
    }
    if (exists) {
        std::string msg = std::string(label) + ": element with index " + std::to_string(index) +
                          " already exists";
        num::debug::panic("DuplicateKeyError", msg, loc);
    }
}

/// @brief Sample the equivalence-relation axioms a union-find claims to maintain.
///
/// `find`, `unite` and `connected` are an *implementation* of an equivalence
/// relation, and the properties that make them correct are reflexivity, symmetry
/// and transitivity of \f$\sim\f$, plus agreement between `connected(u,v)` and
/// \f$\mathrm{find}(u) = \mathrm{find}(v)\f$. None of that is decidable from the
/// type, so it is sampled here.
/// Takes a mutable reference because path compression rewrites the forest while
/// answering queries: `find` is observationally pure but not physically const.
template <class DS, std::integral Index = num::idx>
inline void
verify_equivalence_relation(DS &ds, Index n,
                            std::source_location loc = std::source_location::current()) {
    if constexpr (!num::debug::sampling_compiled_in) {
        return;
    }
    if (num::debug::get_level() != num::debug::diagnostic_level::full || n == 0) {
        return;
    }
    const Index limit = n < Index(24) ? n : Index(24);

    for (Index u = 0; u < limit; ++u) {
        // Reflexivity: u ~ u.
        if (!ds.connected(u, u)) {
            num::debug::panic("PropertyError",
                              "union-find is not reflexive: connected(" + std::to_string(u) + "," +
                                  std::to_string(u) + ") is false",
                              loc);
        }
        // find selects a canonical representative, so it is idempotent.
        if (ds.find(ds.find(u)) != ds.find(u)) {
            num::debug::panic("PropertyError",
                              "union-find representative is not canonical: find(find(" +
                                  std::to_string(u) + ")) != find(" + std::to_string(u) + ")",
                              loc);
        }
    }

    for (Index u = 0; u < limit; ++u) {
        for (Index v = 0; v < limit; ++v) {
            // Symmetry: u ~ v iff v ~ u.
            if (ds.connected(u, v) != ds.connected(v, u)) {
                num::debug::panic("PropertyError",
                                  "union-find is not symmetric at (" + std::to_string(u) + "," +
                                      std::to_string(v) + ")",
                                  loc);
            }
            // connected must agree with equality of representatives.
            if (ds.connected(u, v) != (ds.find(u) == ds.find(v))) {
                num::debug::panic("PropertyError",
                                  "union-find connected() disagrees with find() at (" +
                                      std::to_string(u) + "," + std::to_string(v) + ")",
                                  loc);
            }
            if (!ds.connected(u, v)) {
                continue;
            }
            // Transitivity: u ~ v and v ~ w imply u ~ w.
            for (Index w = 0; w < limit; ++w) {
                if (ds.connected(v, w) && !ds.connected(u, w)) {
                    num::debug::panic("PropertyError",
                                      "union-find is not transitive: " + std::to_string(u) + "~" +
                                          std::to_string(v) + " and " + std::to_string(v) + "~" +
                                          std::to_string(w) + " but not " + std::to_string(u) +
                                          "~" + std::to_string(w),
                                      loc);
                }
            }
        }
    }

    if (ds.count() > ds.size()) {
        num::debug::panic("PropertyError",
                          "union-find quotient is larger than the underlying set: count() = " +
                              std::to_string(ds.count()) +
                              " > size() = " + std::to_string(ds.size()),
                          loc);
    }
}

/// @brief Verify the heap axiom: the reported top key is minimal over the queue.
template <class PQ, class Key = double, std::integral Index = num::idx>
inline void verify_heap_order(const PQ &pq, Index capacity,
                              std::source_location loc = std::source_location::current()) {
    if (num::debug::get_level() != num::debug::diagnostic_level::full || pq.empty()) {
        return;
    }
    const Key top = pq.top_key();
    for (Index i = 0; i < capacity; ++i) {
        if (pq.contains(i) && pq.key_of(i) < top) {
            num::debug::panic("PropertyError",
                              "priority queue violates the heap property: element " +
                                  std::to_string(i) + " has a key smaller than top_key()",
                              loc);
        }
    }
}

/// @brief Verify vertex index is within graph vertex range [0, n_vertices).
template <std::integral Index>
inline void check_vertex_bounds(Index u, Index n_vertices, std::string_view label = "graph::vertex",
                                std::source_location loc = std::source_location::current()) {
    if (num::debug::get_level() == num::debug::diagnostic_level::off) {
        return;
    }
    if (u >= n_vertices) {
        std::string msg = std::string(label) + ": vertex index " + std::to_string(u) +
                          " >= graph vertex count " + std::to_string(n_vertices);
        num::debug::panic("VertexOutOfBoundsError", msg, loc);
    }
}

/// @brief Verify edge weight is strictly positive.
template <typename Weight>
inline void check_positive_weight(const Weight &weight,
                                  std::string_view label = "graph::edge_weight",
                                  std::source_location loc = std::source_location::current()) {
    if (num::debug::get_level() == num::debug::diagnostic_level::off) {
        return;
    }
    if (!(weight > Weight{0})) {
        std::string msg = std::string(label) + ": edge weight must be strictly positive";
        num::debug::panic("NonPositiveWeightError", msg, loc);
    }
}

/// @brief Verify graph connectivity invariant (1 connected component).
template <std::integral Index>
inline void check_connected(Index n_components, std::string_view label = "graph",
                            std::source_location loc = std::source_location::current()) {
    if (num::debug::get_level() == num::debug::diagnostic_level::off) {
        return;
    }
    if (n_components > 1) {
        std::string msg = std::string(label) + " is disconnected with " +
                          std::to_string(n_components) + " disjoint components";
        num::debug::panic("DisconnectedGraphError", msg, loc);
    }
}

/// @brief Verify that reported degrees agree with the enumerated adjacency.
///
/// `degree(u)` and `neighbors(u)` are two presentations of the same set
/// \f$\{v : (u,v) \in E\}\f$; a graph whose degrees disagree with its adjacency
/// silently corrupts every traversal built on it.
template <class G, std::integral Index = num::idx>
inline void verify_degree_consistency(const G &g,
                                      std::source_location loc = std::source_location::current()) {
    if (num::debug::get_level() == num::debug::diagnostic_level::off) {
        return;
    }
    for (Index u = 0; u < g.n_vertices(); ++u) {
        Index counted = 0;
        for ([[maybe_unused]] const auto &edge : g.neighbors(u)) {
            ++counted;
        }
        if (counted != g.degree(u)) {
            num::debug::panic("GraphStructureError",
                              "degree(" + std::to_string(u) + ") = " + std::to_string(g.degree(u)) +
                                  " but neighbors(" + std::to_string(u) + ") enumerates " +
                                  std::to_string(counted) + " entries",
                              loc);
        }
    }
}

/// @brief Verify the handshake lemma \f$\sum_u \deg(u) = 2|E|\f$ for an undirected graph.
template <class G, std::integral Index = num::idx>
inline void verify_handshake_lemma(const G &g,
                                   std::source_location loc = std::source_location::current()) {
    if (num::debug::get_level() == num::debug::diagnostic_level::off) {
        return;
    }
    Index degree_sum = 0;
    for (Index u = 0; u < g.n_vertices(); ++u) {
        degree_sum += g.degree(u);
    }
    if (degree_sum != Index(2) * g.n_edges()) {
        num::debug::panic(
            "GraphStructureError",
            "handshake lemma violated: sum of degrees = " + std::to_string(degree_sum) +
                " but 2|E| = " + std::to_string(Index(2) * g.n_edges()) +
                ". The graph is not a consistent undirected incidence structure.",
            loc);
    }
}

/// @brief Verify the defining structure of a graph Laplacian \f$\Delta = D - W\f$.
///
/// A Laplacian is symmetric with zero row sums, which is exactly the statement
/// \f$\Delta \mathbf{1} = 0\f$: the constant vector lies in its null space. That
/// null space is why a Laplacian is positive *semi*-definite and must be asserted
/// `law::psd` rather than `law::spd` — asserting the latter claims an
/// invertibility it does not have.
///
/// Accepts dense or CSR storage; for CSR only the stored entries are visited, so
/// the check stays proportional to the number of non-zeros.
template <class Mat>
inline void verify_laplacian_structure(const Mat &L, double tol = 1e-9,
                                       std::source_location loc = std::source_location::current()) {
    if (num::debug::get_level() == num::debug::diagnostic_level::off) {
        return;
    }

    num::idx n = 0;
    num::idx m = 0;
    if constexpr (requires { L.n_rows(); }) {
        n = L.n_rows();
        m = L.n_cols();
    } else {
        n = L.rows();
        m = L.cols();
    }
    if (n != m) {
        num::debug::panic("GraphStructureError", "Laplacian must be square", loc);
    }

    auto check_row = [&](num::idx i, double row_sum) {
        if (std::abs(row_sum) > tol) {
            num::debug::panic(
                "GraphStructureError",
                "Laplacian row " + std::to_string(i) + " sums to " + std::to_string(row_sum) +
                    " rather than 0, so the constant vector is not in its null space.",
                loc);
        }
    };
    auto check_symmetry = [&](num::idx i, num::idx j) {
        const double a = static_cast<double>(L(i, j));
        const double b = static_cast<double>(L(j, i));
        if (std::abs(a - b) > tol) {
            num::debug::panic("GraphStructureError",
                              "Laplacian is not symmetric at (" + std::to_string(i) + "," +
                                  std::to_string(j) + ")",
                              loc);
        }
    };

    if constexpr (requires {
                      L.row_ptr();
                      L.col_idx();
                      L.values();
                  }) {
        const auto *row_ptr = L.row_ptr();
        const auto *col_idx = L.col_idx();
        const auto *values = L.values();
        for (num::idx i = 0; i < n; ++i) {
            double row_sum = 0.0;
            for (num::idx k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
                row_sum += static_cast<double>(values[k]);
                check_symmetry(i, col_idx[k]);
            }
            check_row(i, row_sum);
        }
    } else {
        for (num::idx i = 0; i < n; ++i) {
            double row_sum = 0.0;
            for (num::idx j = 0; j < n; ++j) {
                row_sum += static_cast<double>(L(i, j));
                check_symmetry(i, j);
            }
            check_row(i, row_sum);
        }
    }
}

} // namespace num::structures::debug
