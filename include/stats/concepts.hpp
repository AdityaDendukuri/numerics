/// @file stats/concepts.hpp
/// @brief Contracts for streaming statistical accumulators.
#pragma once

#include "algebra/concepts.hpp"
#include "core/types.hpp"
#include <concepts>

namespace num {

/// @brief Accumulator absorbing samples one at a time and reporting a summary.
///
/// The defining property is that cost per sample and storage are both constant.
/// Welford's recurrence and a fixed-bin histogram qualify, while anything that
/// retains the samples in order to summarize them does not.
template <class A, class T = real>
concept streaming_accumulator = scalars::field<T> && requires(A acc, T x) {
    acc.update(x);
    { acc.count } -> std::convertible_to<idx>;
};

/// @brief Accumulator reporting the first two moments.
///
/// \f[
/// \mu_n = \mu_{n-1} + \frac{x_n - \mu_{n-1}}{n}, \qquad
/// s^2 = \frac{M_{2,n}}{n - 1}
/// \f]
///
/// The recurrence is used rather than \f$\sum x^2 - n\mu^2\f$ because the latter
/// cancels catastrophically when the variance is small relative to the mean.
template <class A, class T = real>
concept moment_accumulator = streaming_accumulator<A, T> && requires(const A &acc) {
    { acc.variance() } -> std::convertible_to<T>;
    { acc.std_dev() } -> std::convertible_to<T>;
};

} // namespace num
