/// @file fluid/particle3d.hpp
/// @brief 3D SPH particle type
#pragma once

namespace physics {

/// @brief 3D SPH particle  -- AoS layout
struct Particle3D {
    float x, y, z;
    float vx, vy, vz;
    float evx, evy, evz; ///< smoothed velocity (ev = (ev+v)/2 each step)
    float ax, ay, az;
    float density, pressure;
    float temperature, dT_dt;
};

} // namespace physics
