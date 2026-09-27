/// @file spectral/concepts.hpp
/// @brief Contracts for discrete transforms.
#pragma once

#include "algebra/concepts.hpp"
#include "container/vector.hpp"
#include "core/types.hpp"
#include <concepts>

namespace num {

/// @brief Reusable plan executing a fixed-length transform.
///
/// \f[ X_k = \sum_{n=0}^{N-1} x_n \, e^{-2\pi i k n / N} \f]
///
/// A plan is built once for a length and reused, which is what lets a backend
/// precompute twiddle factors or hand the length to FFTW. Transform length is
/// fixed at construction, so a plan and a vector must agree on it.
template <class P, class V = cvec>
concept transform_plan = inner_product_space<V> && requires(const P &plan, const V &in, V &out) {
    { plan.size() } -> std::convertible_to<int>;
    plan.execute(in, out);
};

} // namespace num
