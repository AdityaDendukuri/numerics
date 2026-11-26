/// @file structures/concepts.hpp
/// @brief Concepts for discrete structures: union-find, priority queues and graphs.
#pragma once

#include "algebra/scalar.hpp"
#include "core/types.hpp"
#include <concepts>
#include <span>
#include <type_traits>

namespace num::concepts {

/// @brief Structure that keeps an equivalence relation \f$\sim\f$ on \f$\{0, \ldots, N-1\}\f$
/// (union-find).
template <typename T, typename Index = num::idx>
concept equivalence_relation = requires(T ds, Index u, Index v) {
    { ds.find(u) } -> std::same_as<Index>;
    { ds.unite(u, v) } -> std::same_as<bool>;
    { ds.connected(u, v) } -> std::same_as<bool>;
    { ds.count() } -> std::same_as<Index>;
    { ds.size() } -> std::same_as<Index>;
};

/// @brief Priority queue whose keys are addressed by index, with a minimum and in-place `update`.
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

/// @brief Edge \f$u \to v\f$ with weight \f$w\f$.
template <typename E, typename Index = num::idx, typename Weight = double>
concept weighted_incidence = field<Weight> && requires(E e) {
    { e.to } -> std::convertible_to<Index>;
    { e.weight } -> std::convertible_to<Weight>;
};

/// @brief Graph \f$G = (V, E)\f$ with `neighbors` and `degree`.
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
