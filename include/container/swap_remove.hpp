/// @file container/swap_remove.hpp
/// @brief Removal by swapping in the tail, with index remapping.
#pragma once

#include "container/vector.hpp"
#include "core/types.hpp"
#include <stdexcept>
#include <utility>

namespace num {

/// @brief One move: the element at `from` goes to `to`.
struct index_move {
    idx from;
    idx to;
};

/// @brief Plan the moves that fill each removed slot from the end of the array.
/// `removed` must be distinct.
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

/// @brief Remove the elements at `removed` by moving elements from the end into their places.
/// @warning Moved elements change index; use `swap_remove_map` to repair indices held elsewhere.
template <typename T> void swap_remove(array<T> &values, view<const idx> removed) {
    for (const index_move &move : swap_remove_plan(values.size(), removed)) {
        values[move.to] = std::move(values[move.from]);
    }
    values.resize(values.size() - removed.size());
}

/// @brief Remove the elements at `removed`, keeping the others in order.
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
            if (kept != index) {
                values[kept] = std::move(values[index]);
            }
            ++kept;
        }
    }
    values.resize(kept);
}

/// @brief Index reported for a removed element by `swap_remove_map`.
inline constexpr idx removed_index = static_cast<idx>(-1);

/// @brief Map each old index to its new one after a swap-remove, or to `removed_index`.
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
