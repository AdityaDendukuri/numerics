/// @file heat.hpp
/// @brief Heat transfer parameters for the SPH backends.
///
/// HeatParams is consumed by backends::seq::heat_compute and
/// backends::omp::heat_compute.  The dispatch happens in fluid.cpp.
#pragma once

namespace physics {

struct HeatParams {
    float h        = 0.025f;  ///< Smoothing length [m]
    float alpha_T  = 0.005f;  ///< Thermal diffusivity [m^2/s]
    float h_conv   = 8.0f;    ///< Convective heat transfer coefficient [1/s]
    float mass     = 0.4f;    ///< Particle mass [kg]
};

} // namespace physics
