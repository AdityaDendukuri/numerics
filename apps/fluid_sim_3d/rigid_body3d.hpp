/// @file fluid/rigid_body3d.hpp
/// @brief 3D rigid body type for SPH boundary interactions
#pragma once

namespace physics {

/// @brief 3D rigid spherical body that interacts with SPH particles
struct RigidBody3D {
    float x, y, z;
    float vx, vy, vz;
    float radius;
    float temperature;
    float mass;
    bool  fixed;
    bool  is_heat_source;
};

} // namespace physics
