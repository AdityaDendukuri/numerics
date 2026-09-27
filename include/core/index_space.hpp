/// @file core/index_space.hpp
/// @brief Discrete index spaces: the root the grid and lattice families are built on.
///
/// A grid or lattice is a finite set of sites with a linear index, not a vector space. A field
/// sampled on it is the vector space, so one grid can carry real, complex or several fields.
#pragma once

#include <concepts>

namespace num {

/// @brief A square \f$N \times N\f$ extent.
///
/// A structured grid and a periodic lattice share exactly this much structure: both are
/// square and know their side length. Naming it makes the two families siblings rather
/// than independent shapes that happen to spell `N` the same way.
template <class G>
concept square_extent_2d = requires(const G &g) {
    { g.N } -> std::convertible_to<int>;
};

/// @brief A square extent carrying a flattening \f$\mathbb{Z}^2 \to \mathbb{Z}\f$.
///
/// `flat(i, j)` puts a site at a linear position, which is what lets a field over the grid
/// be stored as one contiguous vector and handed to `num::kernel` unchanged.
template <class G>
concept cartesian_index_space_2d = square_extent_2d<G> && requires(const G &g, int i, int j) {
    { g.flat(i, j) } -> std::convertible_to<int>;
};

/// @brief A square extent carrying precomputed periodic neighbour maps.
///
/// The wrap-around is folded into the four index arrays. A stencil sweep therefore needs
/// no modulo and no boundary branch, which is why these are stored rather than computed.
template <class P>
concept periodic_neighbourhood_2d = square_extent_2d<P> && requires(const P &lattice, int i) {
    { lattice.up[i] } -> std::convertible_to<int>;
    { lattice.dn[i] } -> std::convertible_to<int>;
    { lattice.lt[i] } -> std::convertible_to<int>;
    { lattice.rt[i] } -> std::convertible_to<int>;
};

} // namespace num
