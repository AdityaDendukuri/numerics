/// @file fluid3d.hpp
/// @brief 3D WCSPH fluid solver  -- public interface and dispatch hub
///
/// FluidSolver3D is a data container + dispatch hub, mirroring fluid.hpp.
/// All physics computation lives in backends/{seq,omp}/.
///
/// @par Execution policies (params.policy)
///   Backend::seq   -- Newton pair traversal via iterate_pairs(), O(n*k/2)
///   Backend::omp   -- parallel for over particles, per-particle query, O(n*k)
#pragma once

#include "particle3d.hpp"
#include "rigid_body3d.hpp"
#include "spatial_hash3d.hpp"
#include "heat3d.hpp"
#include "core/policy.hpp"
#include <vector>

namespace physics {

using num::Backend;

struct FluidParams3D {
    float h     = 0.05f;   ///< Smoothing length [m]
    float rho0  = 1000.0f; ///< Rest density [kg/m^3]
    int   gamma = 7;       ///< Tait EOS exponent
    float c0    = 10.0f;   ///< Speed of sound [m/s]
    float mu    = 10.0f;   ///< Dynamic viscosity [Pa*s]
    float mass  = 0.064f;  ///< Particle mass [kg]  (~= rho_0*(0.8h)^3)

    float gx = 0.0f, gy = -9.81f, gz = 0.0f;
    float dt = 0.001f;

    float xmin = 0.0f, xmax = 0.8f;
    float ymin = 0.0f, ymax = 0.8f;
    float zmin = 0.0f, zmax = 0.8f;
    float restitution = 0.01f;

    float alpha_T = 0.005f; ///< Thermal diffusivity [m^2/s]
    float h_conv  = 8.0f;   ///< Convective coefficient with rigid bodies [1/s]

    Backend policy = Backend::seq; ///< seq = Newton pairs; omp = parallel
};

class FluidSolver3D {
  public:
    explicit FluidSolver3D(const FluidParams3D& p);

    void add_particle(float x,
                      float y,
                      float z,
                      float vx,
                      float vy,
                      float vz,
                      float T);
    void add_body(const RigidBody3D& b);
    void clear();

    /// Advance by one timestep.  Dispatches to seq or omp backends.
    void step();

    const std::vector<Particle3D>& particles() const {
        return particles_;
    }
    const std::vector<RigidBody3D>& bodies() const {
        return bodies_;
    }
    std::vector<RigidBody3D>& bodies() {
        return bodies_;
    }
    const FluidParams3D& params() const {
        return params_;
    }
    FluidParams3D& params_mut() {
        return params_;
    }
    float min_temp() const {
        return T_min_;
    }
    float max_temp() const {
        return T_max_;
    }

  private:
    FluidParams3D            params_;
    std::vector<Particle3D>  particles_;
    std::vector<RigidBody3D> bodies_;
    SpatialHash3D            grid_;
    float                    T_min_ = 0.0f, T_max_ = 100.0f;
};

} // namespace physics
