/// @file stats/concepts.hpp
/// @brief Concepts for streaming statistics.
#pragma once

#include "algebra/concepts.hpp"
#include "core/types.hpp"
#include <concepts>

namespace num {

/// @brief Streaming accumulator: constant cost and storage per sample.
template <class A, class T = real>
concept streaming_accumulator = field<T> && requires(A acc, T x) {
    acc.update(x);
    { acc.count } -> std::convertible_to<idx>;
};

/// @brief Moment accumulator: mean \f$\mu\f$ and variance \f$s^2\f$ by Welford's recurrence.
template <class A, class T = real>
concept moment_accumulator = streaming_accumulator<A, T> && requires(const A &acc) {
    { acc.variance() } -> std::convertible_to<T>;
    { acc.std_dev() } -> std::convertible_to<T>;
};

} // namespace num
