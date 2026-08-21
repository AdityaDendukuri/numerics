/// @file stats/selection.hpp
/// @brief Index selection by scalar score.
#pragma once

#include "core/types.hpp"
#include <algorithm>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

namespace num {

/// Return the first index with the largest projected value.
template<typename Score>
[[nodiscard]] idx argmax(idx count, Score&& score) {
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
template<typename T>
[[nodiscard]] idx argmax(std::span<const T> values) {
  return argmax(values.size(), [&](idx index) -> const T& { return values[index]; });
}

/// Return indices of the k smallest values, ordered by increasing value.
template<typename T>
[[nodiscard]] std::vector<idx> smallest_indices(std::span<const T> values, idx count) {
  count = std::min(count, values.size());
  std::vector<idx> indices(values.size());
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

} // namespace num
