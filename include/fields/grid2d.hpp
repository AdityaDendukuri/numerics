/// @file fields/grid2d.hpp
/// @brief 2D uniform interior grid on \f$[0, 1]^2\f$.
#pragma once

#include "core/index_space.hpp"
#include "core/types.hpp"

namespace num {

/// @brief Uniform grid of \f$N \times N\f$ interior points, \f$h = 1 / (N + 1)\f$.
struct grid2d {
    int N;    ///< interior nodes per side
    double h; ///< grid spacing = 1/(N+1)

    [[nodiscard]] double x(int i) const { return (i + 1) * h; }
    [[nodiscard]] double y(int j) const { return (j + 1) * h; }
    [[nodiscard]] int flat(int i, int j) const { return (i * N) + j; }
    [[nodiscard]] int size() const { return N * N; }
};

static_assert(cartesian_index_space_2d<grid2d>);
} // namespace num
