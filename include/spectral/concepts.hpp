/// @file spectral/concepts.hpp
/// @brief Concepts for discrete transforms.
#pragma once

#include "algebra/concepts.hpp"
#include "container/vector.hpp"
#include "core/types.hpp"
#include <concepts>

namespace num {

/// @brief Transform plan for a fixed length: \f$X_k = \sum_n x_n e^{-2\pi i k n / N}\f$.
template <class P, class V = vec<cplx>>
concept transform_plan = inner_product_space<V> && requires(const P &plan, const V &in, V &out) {
    { plan.size() } -> std::convertible_to<int>;
    plan.execute(in, out);
};

} // namespace num
