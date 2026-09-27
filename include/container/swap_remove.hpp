/// @file container/swap_remove.hpp
/// @brief Unordered removal that keeps indices contiguous.
#pragma once

#include "container/vector.hpp"
#include "core/types.hpp"
#include <stdexcept>
#include <utility>

namespace num {

/// One relocation performed by a swap-remove: the element at `from` lands at `to`.
struct index_move {
    idx from;
    idx to;
};

/// @brief Plan the relocations that compact `n` elements after removing some.
///
/// Each hole is filled from the tail, which is O(1) per removal but renames the moved
/// elements. The returned renames let parallel arrays and held indices follow. Holes and
/// survivors are taken in ascending order, so the plan ignores the order of `removed`.
///
/// @param n Current number of elements.
/// @param removed Indices to remove. Must be distinct and less than `n`.
/// @return One move per hole that a survivor fills; empty when the removed
///         indices are exactly the tail.
/// @throws std::out_of_range If an index is not less than `n`.
[[nodiscard]] inline array<index_move> swap_remove_plan(idx n, view<const idx> removed) {
    array<bool> gone(n, false);
    for (idx index : removed) {
        if (index >= n) {
            throw std::out_of_range("swap_remove_plan: index is outside the range");
        }
        gone[index] = true;
    }
    const idx kept = n - removed.size();
    array<idx> holes, movers;
    for (idx index = 0; index < kept; ++index) {
        if (gone[index]) {
            holes.push_back(index);
        }
    }
    for (idx index = kept; index < n; ++index) {
        if (!gone[index]) {
            movers.push_back(index);
        }
    }
    array<index_move> plan;
    for (idx k = 0; k < holes.size(); ++k) {
        plan.push_back({movers[k], holes[k]});
    }
    return plan;
}

/// @brief Apply `swap_remove_plan` to one array, shrinking it in place.
///
/// Elements are moved, not copied, so this is as cheap for a container element
/// as for a scalar one.
template <typename T> void swap_remove(array<T> &values, view<const idx> removed) {
    for (const index_move &move : swap_remove_plan(values.size(), removed)) {
        values[move.to] = std::move(values[move.from]);
    }
    values.resize(values.size() - removed.size());
}

/// @brief Remove the elements at `removed` and keep the others in their order.
///
/// The order-preserving counterpart of `swap_remove`: it costs one pass over
/// the array instead of one move per removal.
///
/// @throws std::out_of_range If an index is not less than `values.size()`.
template <typename T> void remove_indices(array<T> &values, view<const idx> removed) {
    array<bool> gone(values.size(), false);
    for (idx index : removed) {
        if (index >= values.size()) {
            throw std::out_of_range("remove_indices: index is outside the range");
        }
        gone[index] = true;
    }
    idx kept = 0;
    for (idx index = 0; index < values.size(); ++index) {
        if (!gone[index]) {
            values[kept++] = std::move(values[index]);
        }
    }
    values.resize(kept);
}

/// The index a removed element reports through `swap_remove_map`.
inline constexpr idx removed_index = static_cast<idx>(-1);

/// @brief Map every old index to its new one, or to `removed_index`.
///
/// Use this to repair indices held outside the arrays being compacted, such as
/// the position a walker occupies or an entry in a lookup table.
[[nodiscard]] inline array<idx> swap_remove_map(idx n, view<const idx> removed) {
    array<idx> mapping(n);
    for (idx index = 0; index < n; ++index) {
        mapping[index] = index;
    }
    for (idx index : removed) {
        mapping[index] = removed_index;
    }
    for (const index_move &move : swap_remove_plan(n, removed)) {
        mapping[move.from] = move.to;
    }
    return mapping;
}

} // namespace num
