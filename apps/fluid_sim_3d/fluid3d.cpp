/// @file fluid3d.cpp
/// @brief Single-switch dispatch from Backend enum to 3D SPH backend
/// implementations.
///
/// Mirrors apps/fluid_sim/fluid.cpp exactly:
///   - Includes both seq and omp impl.hpp headers
///   - switch(params_.policy) routes each phase to the correct backend
///   - Non-computational methods (add_particle, clear, ...) live here directly
///
/// Adding a new backend:
///   1. Add the enumerator to num::Backend in include/core/policy.hpp
///   2. Create apps/fluid_sim_3d/backends/<name>/ with impl.hpp + fluid.cpp +
///   heat.cpp
///   3. Add `case num::Backend::<name>:` to the switch in step()
///   4. Register the .cpp files in apps/fluid_sim_3d/CMakeLists.txt

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
