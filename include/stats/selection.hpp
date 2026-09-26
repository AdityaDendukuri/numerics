/// @file stats/selection.hpp
/// @brief Index selection by scalar score.
#pragma once

#include "core/types.hpp"
#include <algorithm>
#include <numeric>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace num {

/// Return the first index with the largest projected value.
template <typename Score>
[[nodiscard]] idx argmax(idx count, Score &&score) {
    if (count == 0) {
        throw std::invalid_argument("argmax: empty range");
    }
    idx best = 0;
    auto best_value = score(best);
    for (idx index = 1; index < count; ++index) {
        auto value = score(index);
        if (value > best_value) {
            best = index;
            best_value = value;
        }
    }
    return best;
}

/// Return the first index of the largest value.
template <typename T>
[[nodiscard]] idx argmax(view<const T> values) {
    return argmax(values.size(), [&](idx index) -> const T & { return values[index]; });
}

/// Return the indices that sort `values` in increasing order, ties by index.
///
/// `values` is any container with `size()` and `operator[]`.
template <typename Values>
[[nodiscard]] array<idx> argsort(const Values &values) {
    array<idx> indices(values.size());
    std::iota(indices.begin(), indices.end(), idx{0});
    std::stable_sort(indices.begin(), indices.end(),
                     [&](idx left, idx right) { return values[left] < values[right]; });
    return indices;
}

/// Return the first `count` indices of `argsort(values)`, without sorting the rest.
template <typename Values>
[[nodiscard]] array<idx> smallest_indices(const Values &values, idx count) {
    count = std::min(count, static_cast<idx>(values.size()));
    array<idx> indices(values.size());
    std::iota(indices.begin(), indices.end(), idx{0});
    const auto less = [&](idx left, idx right) {
        if (values[left] == values[right]) {
            return left < right;
        }
        return values[left] < values[right];
    };
    if (count < indices.size()) {
        std::nth_element(indices.begin(), indices.begin() + count, indices.end(), less);
        indices.resize(count);
    }
    std::sort(indices.begin(), indices.end(), less);
    return indices;
}

/// Return the indices in [0, count) for which `keep(index)` is true, in increasing order.
template <typename Predicate>
[[nodiscard]] array<idx> filter(idx count, Predicate &&keep) {
    array<idx> indices;
    for (idx index = 0; index < count; ++index) {
        if (keep(index)) {
            indices.push_back(index);
        }
    }
    return indices;
}

/// @brief Group the indices in [0, count) by `key(index)`.
///
/// Returns the distinct keys in order of first appearance, and for each index
/// the position of its key among them.
template <typename Key>
[[nodiscard]] auto group_by(idx count, Key &&key) {
    using value = std::remove_cvref_t<decltype(key(idx{0}))>;
    std::pair<array<value>, array<idx>> groups;
    auto &[keys, group_of] = groups;
    unordered_map<value, idx> position;
    group_of.reserve(count);
    for (idx index = 0; index < count; ++index) {
        const auto [entry, added] = position.emplace(key(index), keys.size());
        if (added) {
            keys.push_back(entry->first);
        }
        group_of.push_back(entry->second);
    }
    return groups;
}

} // namespace num
