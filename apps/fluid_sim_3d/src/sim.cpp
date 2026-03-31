/// @file src/sim.cpp
/// @brief Single-switch dispatch from Backend enum to 3D SPH backend
/// implementations.
///
/// Moved from root fluid3d.cpp.  Listed as the compiled unit in the new
/// CMakeLists so that include/ paths resolve correctly under the restructured
/// layout.
///
/// Mirrors apps/fluid_sim/src/fluid.cpp exactly for the 3D case.

#include "fluid3d.hpp"
#include "backends/seq/impl.hpp"
#include "backends/omp/impl.hpp"

namespace physics {

FluidSolver3D::FluidSolver3D(const FluidParams3D& p)
    : params_(p)
    , grid_(2.0f * p.h, p.xmin, p.xmax, p.ymin, p.ymax, p.zmin, p.zmax) {}

void FluidSolver3D::add_particle(float x,
                                 float y,
                                 float z,
                                 float vx,
                                 float vy,
                                 float vz,
                                 float T) {
    Particle3D p{};
    p.x           = x;
    p.y           = y;
    p.z           = z;
    p.vx          = vx;
    p.vy          = vy;
    p.vz          = vz;
    p.evx         = vx;
    p.evy         = vy;
    p.evz         = vz;
    p.density     = params_.rho0;
    p.pressure    = 0.0f;
    p.temperature = T;
    p.dT_dt       = 0.0f;
    particles_.push_back(p);
}

void FluidSolver3D::add_body(const RigidBody3D& b) {
    bodies_.push_back(b);
}
void FluidSolver3D::clear() {
    particles_.clear();
    bodies_.clear();
}

// Main step  -- policy dispatch

void FluidSolver3D::step() {
    if (particles_.empty())
        return;

    for (Particle3D& p : particles_) {
        p.ax = p.ay = p.az = 0.0f;
        p.dT_dt            = 0.0f;
    }

    grid_.build(particles_); // O(n + C) counting sort  -- always sequential

    const HeatParams3D hp{params_.h,
                          params_.alpha_T,
                          params_.h_conv,
                          params_.mass};

    switch (params_.policy) {
        case num::Backend::omp:
            backends::omp::compute_density_pressure(particles_, params_, grid_);
            backends::omp::compute_forces(particles_, params_, grid_);
            backends::omp::heat_compute(particles_, bodies_, grid_, hp);
            backends::omp::body_collisions(particles_, bodies_, params_);
            backends::omp::integrate(particles_, params_);
            backends::omp::enforce_boundaries(particles_, params_);
            backends::omp::update_temp_range(particles_,
                                             bodies_,
                                             T_min_,
                                             T_max_);
            backends::seq::integrate_bodies(bodies_,
                                            params_); // M bodies tiny  -- seq
            break;

        default: // seq: Newton pair traversal
            backends::seq::compute_density_pressure(particles_, params_, grid_);
            backends::seq::compute_forces(particles_, params_, grid_);
            backends::seq::heat_compute(particles_, bodies_, grid_, hp);
            backends::seq::body_collisions(particles_, bodies_, params_);
            backends::seq::integrate(particles_, params_);
            backends::seq::enforce_boundaries(particles_, params_);
            backends::seq::update_temp_range(particles_,
                                             bodies_,
                                             T_min_,
                                             T_max_);
            backends::seq::integrate_bodies(bodies_, params_);
            break;
    }
}

} // namespace physics
