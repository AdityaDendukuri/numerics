/// @file fluid.hpp
/// @brief Weakly Compressible SPH (WCSPH) fluid solver  -- public interface
///
/// FluidSolver is a data container + dispatch hub.  All physics computation
/// lives in apps/fluid_sim/backends/{seq,omp}/, mirroring the structure of
/// src/backends/{seq,omp}/ in the numerics library.
///
/// @par Execution policies (params.policy)
///
///   Backend::seq  (default)
///     Single-threaded.  Newton's 3rd law pair traversal (iterate_pairs)
///     visits each unique {i,j} pair once -> O(n*k/2).
///
///   Backend::omp
///     OpenMP parallel for over particles.  Per-particle neighbour query
///     so each thread writes only to particles_[i]  -- no atomics needed.
///     Falls back to seq when NUMERICS_HAS_OMP is not defined.
///
/// Policy dispatch in step() mirrors src/backends/dispatch.cpp:
///   switch (params_.policy) { case Backend::omp: backends::omp::...; }
#pragma once

#include "particle.hpp"
#include "rigid_body.hpp"
#include "spatial_hash.hpp"
#include "heat.hpp"
#include "core/policy.hpp"
#include <vector>

namespace physics {

using num::Backend;

struct FluidParams {
    // SPH kernel
    float h = 0.025f; ///< Smoothing length [m]

    // Fluid properties
    float rho0  = 1000.0f; ///< Rest density [kg/m^3]
    int   gamma = 7;       ///< Tait EOS exponent
    float c0    = 10.0f;   ///< Speed of sound [m/s]  -> B = rho_0c_0^2/gamma
    float mu    = 0.01f;   ///< Dynamic viscosity [Pa*s]
    float mass  = 0.4f;    ///< Particle mass [kg]  (~= rho_0*dx^2, dx=0.8h)

    // Body forces
    float gx = 0.0f;   ///< Gravity x [m/s^2]
    float gy = -9.81f; ///< Gravity y [m/s^2]

    // Time integration
    float dt = 0.001f; ///< Timestep [s]  (must satisfy CFL: dt < h/c0)

    // Domain
    float xmin        = 0.0f;
    float xmax        = 1.0f;
    float ymin        = 0.0f;
    float ymax        = 0.7f;
    float restitution = 0.3f; ///< Velocity restitution at walls

    // Thermal
    float alpha_T = 0.005f; ///< Thermal diffusivity [m^2/s]
    float h_conv  = 8.0f;   ///< Convective coefficient with rigid bodies [1/s]

    // Execution policy  -- dispatched in FluidSolver::step()
    Backend policy = Backend::seq; ///< seq = Newton pairs; omp = parallel
};

class FluidSolver {
  public:
    explicit FluidSolver(const FluidParams& params);

    void add_particle(float x, float y, float vx, float vy, float temperature);
    void add_body(const RigidBody& body);
    void clear();

    /// Advance the simulation by one timestep (params_.dt).
    /// Dispatches to backends::seq or backends::omp based on params_.policy.
    void step();

    const std::vector<Particle>& particles() const {
        return particles_;
    }
    const std::vector<RigidBody>& bodies() const {
        return bodies_;
    }
    std::vector<RigidBody>& bodies() {
        return bodies_;
    }
    const FluidParams& params() const {
        return params_;
    }

    float min_temp() const {
        return T_min_;
    }
    float max_temp() const {
        return T_max_;
    }

  private:
    FluidParams            params_;
    std::vector<Particle>  particles_;
    std::vector<RigidBody> bodies_;
    SpatialHash            grid_; ///< CellList2D-backed, rebuilt each step

    float T_min_ = 0.0f;
    float T_max_ = 100.0f;
};

} // namespace physics
