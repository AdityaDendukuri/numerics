/// @file core/index_space.hpp
/// @brief Discrete index spaces: square grids and periodic lattices.
#pragma once

#include <concepts>

namespace num {

/// @brief Square index space: an \f$N \times N\f$ extent.
template <class G>
concept square_extent_2d = requires(const G &g) {
    { g.N } -> std::convertible_to<int>;
};

/// @brief Square index space that numbers its sites: \f$(i, j) \mapsto \mathrm{flat}(i, j)\f$.
template <class G>
concept cartesian_index_space_2d = square_extent_2d<G> && requires(const G &g, int i, int j) {
    { g.flat(i, j) } -> std::convertible_to<int>;
};

/// @brief Periodic square index space that stores each site's neighbours.
template <class P>
concept periodic_neighbourhood_2d = square_extent_2d<P> && requires(const P &lattice, int i) {
    { lattice.up[i] } -> std::convertible_to<int>;
    { lattice.dn[i] } -> std::convertible_to<int>;
    { lattice.lt[i] } -> std::convertible_to<int>;
    { lattice.rt[i] } -> std::convertible_to<int>;
};

} // namespace num
