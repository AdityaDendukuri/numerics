/// @file fields/concepts.hpp
/// @brief Contracts for grids and the fields sampled on them.
#pragma once

#include "algebra/concepts.hpp"
#include "container/vector.hpp"
#include "core/types.hpp"
#include <concepts>

namespace num {

/// @brief Rectangular index space with a flattening from coordinates to storage.
///
/// The grid holds the shape and the spacing. It owns no values, so several fields
/// may share one grid.
template <class G>
concept StructuredGrid2D = requires(const G &grid, int i, int j) {
    { grid.N } -> std::convertible_to<int>;
    { grid.flat(i, j) } -> std::convertible_to<int>;
};

/// @brief Scalar-valued field sampled on a grid.
///
/// Indexing by grid coordinate reads a sample. The values are contiguous and
/// exposed as a vector space, so a field can be handed to a Krylov solver without
/// being copied.
template <class F, class T = real>
concept ScalarFieldLike = scalars::Field<T> && requires(F &f, const F &cf, int i, int j) {
    { cf(i, j) } -> std::convertible_to<T>;
    { f(i, j) } -> std::convertible_to<T &>;
    { f.vec() };
};

/// @brief Field whose storage satisfies the vector space contract.
///
/// This is what lets an implicit time stepper solve on a field directly.
template <class F, class T = real>
concept SolvableField = ScalarFieldLike<F, T> && requires(F &f) {
    { f.vec() } -> std::convertible_to<Vector &>;
};

} // namespace num
