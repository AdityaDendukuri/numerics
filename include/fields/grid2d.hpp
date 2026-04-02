/// @file fields/grid2d.hpp
/// @brief 2D uniform interior grid: geometry only, no field data.
///
/// Grid2D describes the spatial discretization of [0,1]^2 into N x N
/// interior nodes with spacing h = 1/(N+1).  It carries no field values
/// and no boundary conditions -- those belong to the operator and the field.
#pragma once

#include "core/types.hpp"

namespace num {

struct Grid2D {
    int    N; ///< interior nodes per side
    double h; ///< grid spacing = 1/(N+1)

    double x(int i) const { return (i + 1) * h; }
    double y(int j) const { return (j + 1) * h; }
    int    flat(int i, int j) const { return i * N + j; }
    int    size() const { return N * N; }
};

} // namespace num
