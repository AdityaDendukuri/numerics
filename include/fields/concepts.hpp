/// @file fields/concepts.hpp
/// @brief Concepts for grids and fields.
#pragma once

#include "algebra/concepts.hpp"
#include "container/vector.hpp"
#include "core/index_space.hpp"
#include "ode/concepts.hpp"
#include "core/types.hpp"
#include <concepts>

namespace num {

/// @brief Scalar field: samples on a grid, stored contiguously.
template <class F, class T = real>
concept scalar_field_like = field<T> && requires(F &f, const F &cf, int i, int j) {
    { cf(i, j) } -> std::convertible_to<T>;
    { f(i, j) } -> std::convertible_to<T &>;
    { f.as_vec() };
};

/// @brief Vector-space field: a scalar field whose storage is a vector space `V`.
template <class F, class T = real, class V = vec<real>>
concept solvable_field = scalar_field_like<F, T> && vec_field<F, V>;

} // namespace num
