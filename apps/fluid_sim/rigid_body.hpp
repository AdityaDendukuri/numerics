/// @file rigid_body.hpp
/// @brief Rigid body (sphere) for SPH collision and heat exchange
#pragma once

namespace physics {

/// @brief A rigid spherical body that interacts with fluid particles
struct RigidBody {
    float x, y;           ///< Center position [m]
    float vx, vy;         ///< Velocity [m/s] (zero for static bodies)
    float radius;         ///< Sphere radius [m]
    float temperature;    ///< Body temperature [ degC]
    float mass;           ///< Mass [kg] (used if dynamic)
    bool  fixed;          ///< If true, body is immovable
    bool  is_heat_source; ///< If true, body maintains its temperature
};

} // namespace physics
