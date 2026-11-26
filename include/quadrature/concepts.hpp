/// @file quadrature/concepts.hpp
/// @brief Concepts for quadrature rules.
#pragma once

#include "algebra/concepts.hpp"
#include "core/types.hpp"
#include <concepts>

namespace num {

/// @brief Rule for integrating on a complex contour: nodes \f$s_k\f$ and weights \f$w_k\f$.
template <class R, class T = real>
concept contour_rule = field<T> && requires(const R &rule, T t) {
    { rule.nodes(t) };
};

} // namespace num
