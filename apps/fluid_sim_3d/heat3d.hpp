/// @file heat3d.hpp
/// @brief Heat transfer parameters for the 3D SPH backends.
#pragma once

namespace physics {

struct HeatParams3D {
    float h       = 0.05f;  ///< Smoothing length [m]
    float alpha_T = 0.005f; ///< Thermal diffusivity [m^2/s]
    float h_conv  = 8.0f;   ///< Convective heat transfer coefficient [1/s]
    float mass    = 0.064f; ///< Particle mass [kg]
};

} // namespace physics
