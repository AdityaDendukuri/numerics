/// @file fields/grid2d.hpp
/// @brief 2D uniform interior grid: geometry only, no field data.
///
/// grid2d describes the spatial discretization of [0,1]^2 into N x N
/// interior nodes with spacing h = 1/(N+1).  It carries no field values
/// and no boundary conditions -- those belong to the operator and the field.
#pragma once

#include "core/types.hpp"

namespace num {

struct grid2d {
    int N;    ///< interior nodes per side
    double h; ///< grid spacing = 1/(N+1)

    [[nodiscard]] double x(int i) const { return (i + 1) * h; }
    [[nodiscard]] double y(int j) const { return (j + 1) * h; }
    [[nodiscard]] int flat(int i, int j) const { return (i * N) + j; }
    [[nodiscard]] int size() const { return N * N; }
};

} // namespace num
