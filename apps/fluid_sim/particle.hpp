/// @file particle.hpp
/// @brief SPH particle data structure
#pragma once

namespace physics {

/// @brief A single SPH fluid particle (float precision for performance)
struct Particle {
    float x, y;        ///< Position [m]
    float vx, vy;      ///< Velocity [m/s]
    float evx, evy;    ///< Smoothed velocity ev = (ev + v)/2 each step  -- used
                       ///  for viscosity to damp high-frequency oscillations
    float ax, ay;      ///< Acceleration [m/s^2] (updated each step)
    float density;     ///< SPH density rho_i [kg/m^3]
    float pressure;    ///< Pressure p_i [Pa]
    float temperature; ///< Temperature T_i [ degC]
    float dT_dt;       ///< Temperature rate [ degC/s] (updated each step)
};

} // namespace physics
