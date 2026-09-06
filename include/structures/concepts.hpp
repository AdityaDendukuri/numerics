/// @file structures/concepts.hpp
/// @brief Contracts for discrete structures, stated as the algebra they maintain.
///
/// A union-find is a representation of an equivalence relation on
/// \f$\{0,\dots,N-1\}\f$ together with its quotient set. A graph is a finite
/// incidence structure. Naming the structure rather than the method list is what
/// lets the laws be stated, and `num::structures::debug` samples them.
#pragma once

#include "algebra/scalar.hpp"
#include "core/types.hpp"
#include <concepts>
#include <span>
#include <type_traits>

namespace num::concepts {

/// @brief Equivalence relation \f$\sim\f$ on \f$\{0,\dots,N-1\}\f$ with its quotient set.
///
/// `find(u)` selects the canonical representative of \f$[u]\f$ and is therefore
/// idempotent. `connected(u,v)` decides \f$u \sim v\f$ and must agree with
/// \f$\mathrm{find}(u) = \mathrm{find}(v)\f$. `unite(u,v)` replaces \f$\sim\f$ by
/// the finest relation containing it in which \f$u \sim v\f$. `count()` is the
/// size of the quotient and drops by one per successful `unite`.
template <typename T, typename Index = num::idx>
concept equivalence_relation = requires(T ds, Index u, Index v) {
    { ds.find(u) } -> std::same_as<Index>;
    { ds.unite(u, v) } -> std::same_as<bool>;
    { ds.connected(u, v) } -> std::same_as<bool>;
    { ds.count() } -> std::same_as<Index>;
    { ds.size() } -> std::same_as<Index>;
};

/// @brief Totally ordered key set with addressable elements and a retrievable minimum.
///
/// `top_key()` is the minimum over every contained key, maintained across `push`,
/// `pop`, and `update`. `update` revises a key in place, which is what Dijkstra
/// and Prim need and a plain priority queue cannot do.
template <typename T, typename Key = double, typename Index = num::idx>
concept addressable_priority_queue = std::totally_ordered<Key> &&
    requires(T pq, Index index, const Key &key) {
    { pq.push(index, key) } -> std::same_as<void>;
    { pq.pop() } -> std::same_as<void>;
    { pq.top_index() } -> std::same_as<Index>;
    { pq.top_key() } -> std::convertible_to<Key>;
    { pq.contains(index) } -> std::same_as<bool>;
    { pq.update(index, key) } -> std::same_as<void>;
    { pq.empty() } -> std::same_as<bool>;
    { pq.size() } -> std::same_as<Index>;
};

/// @brief Directed incidence \f$u \to v\f$ carrying a weight in a scalar field.
template <typename E, typename Index = num::idx, typename Weight = double>
concept weighted_incidence = scalars::field<Weight> && requires(E e) {
    { e.to } -> std::convertible_to<Index>;
    { e.weight } -> std::convertible_to<Weight>;
};

/// @brief Finite incidence structure \f$G = (V, E)\f$ with \f$E \subseteq V \times V\f$.
///
/// `neighbors(u)` enumerates \f$\{v : (u,v) \in E\}\f$ and `degree(u)` is that
/// set's cardinality. The two must agree, and for an undirected graph the
/// handshake lemma \f$\sum_u \deg(u) = 2|E|\f$ must hold.
template <typename G, typename Index = num::idx>
concept incidence_structure = requires(const G g, Index u) {
    { g.n_vertices() } -> std::same_as<Index>;
    { g.n_edges() } -> std::same_as<Index>;
    { g.degree(u) } -> std::same_as<Index>;
    { g.neighbors(u) };
};

} // namespace num::concepts

namespace num {
using concepts::addressable_priority_queue;
using concepts::equivalence_relation;
using concepts::incidence_structure;
using concepts::weighted_incidence;
} // namespace num
