/// @file ode/concepts.hpp
/// @brief Concepts for initial value problems and their state spaces.
#pragma once

#include "container/concepts.hpp"
#include "ode/types.hpp"
#include <concepts>

namespace num {

/// @brief State in a vector space `V`, as implicit methods need.
template <class T, class V = vec<real>>
concept vec_field = vector_space<V> && requires(T &field) {
    { field.as_vec() } -> std::same_as<V &>;
};

} // namespace num
