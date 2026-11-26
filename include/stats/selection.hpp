/// @file stats/selection.hpp
/// @brief Index selection by score.
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

/// @brief Return the first index of the largest score: \f$\arg\max_i s(i)\f$.
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

/// @brief Return the first index of the largest entry: \f$\arg\max_i x_i\f$.
template <typename T>
[[nodiscard]] idx argmax(view<const T> values) {
    return argmax(values.size(), [&](idx index) -> const T & { return values[index]; });
}

/// @brief Return the permutation that sorts `values` ascending, breaking ties by index.
template <typename Values>
[[nodiscard]] array<idx> argsort(const Values &values) {
    array<idx> indices(values.size());
    std::iota(indices.begin(), indices.end(), idx{0});
    std::stable_sort(indices.begin(), indices.end(),
                     [&](idx left, idx right) { return values[left] < values[right]; });
    return indices;
}

/// @brief Return the first `count` indices of `argsort(values)`.
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

/// @brief Return the indices \f$i < \mathrm{count}\f$ for which `keep(i)` holds, ascending.
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

/// @brief Group indices by key: the distinct keys in order of appearance, and each index's key
/// position.
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
