/// @file fields/concepts.hpp
/// @brief Contracts for grids and the fields sampled on them.
#pragma once

#include "algebra/concepts.hpp"
#include "container/vector.hpp"
#include "core/index_space.hpp"
#include "ode/concepts.hpp"
#include "core/types.hpp"
#include <concepts>

namespace num {

/// @brief Rectangular index space with a flattening from coordinates to storage.
///
/// The grid holds the shape and the spacing. It owns no values, so several fields
/// may share one grid.
template <class G>
concept structured_grid_2d = cartesian_index_space_2d<G>;

/// @brief scalar-valued field sampled on a grid.
///
/// Indexing by grid coordinate reads a sample. The values are contiguous and
/// exposed as a vector space, so a field can be handed to a Krylov solver without
/// being copied.
template <class F, class T = real>
concept scalar_field_like = scalars::field<T> && requires(F &f, const F &cf, int i, int j) {
    { cf(i, j) } -> std::convertible_to<T>;
    { f(i, j) } -> std::convertible_to<T &>;
    { f.as_vec() };
};

/// @brief field whose storage satisfies the vector space contract.
///
/// This is what lets an implicit time stepper solve on a field directly. It is exactly
/// `num::vec_field` restricted to fields sampled on a grid, so it refines that rather than
/// restating the requirement — and, like it, takes the space as a parameter instead of
/// naming `vec`.
template <class F, class T = real, class V = vec>
concept solvable_field = scalar_field_like<F, T> && vec_field<F, V>;

} // namespace num
