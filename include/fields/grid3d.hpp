/// @file fields/grid3d.hpp
/// @brief 3D uniform Cartesian grid: geometry only, no field data.
///
/// Grid3D describes an nx*ny*nz lattice with uniform spacing dx and origin
/// (ox,oy,oz). It carries no field values -- those live in a ScalarField3D --
/// mirroring the geometry/data split of Grid2D and ScalarField2D.
/// Flat layout: idx = k*ny*nx + j*nx + i  (x is fastest, z is slowest).
#pragma once

#include "core/types.hpp"

namespace num {

struct Grid3D {
  int nx, ny, nz;                       ///< nodes per axis
  double dx = 1.0;                      ///< uniform cell size
  double ox = 0.0, oy = 0.0, oz = 0.0;  ///< origin (physical coordinate of node 0)

  int size() const { return nx * ny * nz; }

  idx flat(int i, int j, int k) const {
    return static_cast<idx>((k * ny * nx) + (j * nx) + i);
  }

  double x(int i) const { return ox + (i * dx); }
  double y(int j) const { return oy + (j * dx); }
  double z(int k) const { return oz + (k * dx); }
};

} // namespace num
