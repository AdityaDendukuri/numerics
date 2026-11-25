/// @file fields/grid3d.hpp
/// @brief 3D uniform grid.
#pragma once

#include "core/types.hpp"

namespace num {

/// @brief Uniform 3D grid: `nx x ny x nz`, spacing `dx`; flat index \f$(k n_y + j) n_x + i\f$.
struct grid_3d {
    int nx{}, ny{}, nz{};                ///< nodes per axis
    double dx = 1.0;                     ///< uniform cell size
    double ox = 0.0, oy = 0.0, oz = 0.0; ///< origin (physical coordinate of node 0)

    [[nodiscard]] int size() const { return nx * ny * nz; }

    template <typename Type = idx>
    [[nodiscard]] idx flat(Type i, Type j, Type k) const {
        return static_cast<idx>((k * ny * nx) + (j * nx) + i);
    }

    [[nodiscard]] double x(int i) const { return ox + (i * dx); }
    [[nodiscard]] double y(int j) const { return oy + (j * dx); }
    [[nodiscard]] double z(int k) const { return oz + (k * dx); }
};

} // namespace num
