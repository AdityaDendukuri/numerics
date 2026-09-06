/// @file fields/grid3d.hpp
/// @brief 3D uniform Cartesian grid: geometry only, no field data.
///
/// grid_3d describes an nx*ny*nz lattice with uniform spacing dx and origin
/// (ox,oy,oz). It carries no field values -- those live in a scalar_field_3d --
/// mirroring the geometry/data split of grid2d and scalar_field_2d.
/// Flat layout: idx = k*ny*nx + j*nx + i  (x is fastest, z is slowest).
#pragma once

#include "core/types.hpp"

namespace num {

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
