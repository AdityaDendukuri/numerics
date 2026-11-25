/// @file roots/concepts.hpp
/// @brief Concepts for scalar root finding.
#pragma once

#include "algebra/concepts.hpp"
#include "core/types.hpp"
#include <concepts>

namespace num {

/// @brief Bracketable function: a sign change \f$f(a) f(b) < 0\f$ can be tested.
template <class F, class T = real>
concept bracketable_function = scalar_function<F, T> && std::totally_ordered<T>;

} // namespace num
