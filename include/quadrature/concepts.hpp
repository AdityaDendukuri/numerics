/// @file quadrature/concepts.hpp
/// @brief Concepts for quadrature rules.
#pragma once

#include "algebra/concepts.hpp"
#include "core/types.hpp"
#include <concepts>

namespace num {

/// @brief Contour rule: nodes \f$s_k\f$ and weights \f$w_k\f$ on a complex contour.
template <class R, class T = real>
concept contour_rule = field<T> && requires(const R &rule, T t) {
    { rule.nodes(t) };
};

} // namespace num
