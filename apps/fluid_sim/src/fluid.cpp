/// @file src/fluid.cpp
/// @brief Single-switch dispatch from Backend enum to SPH backend implementations.
///
/// Mirrors the root fluid.cpp.  Listed as a compiled unit in the new CMakeLists
/// so that include/ paths resolve correctly under the restructured layout.
///
/// Dispatching mirrors src/backends/dispatch.cpp:
///   switch (params_.policy) { case Backend::omp: backends::omp::...; }
/// Non-computational methods (add_particle, clear, ...) live here directly.

#include "fluid.hpp"
#include "backends/seq/impl.hpp"
#include "backends/omp/impl.hpp"

namespace physics {

FluidSolver::FluidSolver(const FluidParams& params)
    : params_(params)
    , grid_(2.0f * params.h,
            params.xmin, params.xmax,
            params.ymin, params.ymax)
{}

void FluidSolver::add_particle(float x, float y, float vx, float vy, float temperature) {
    Particle p{};
    p.x = x;  p.y = y;
    p.vx = vx; p.vy = vy;
    p.evx = vx; p.evy = vy;
    p.density     = params_.rho0;
    p.pressure    = 0.0f;
    p.temperature = temperature;
    particles_.push_back(p);
}

void FluidSolver::add_body(const RigidBody& body) { bodies_.push_back(body); }
void FluidSolver::clear() { particles_.clear(); bodies_.clear(); }

// Main step  -- policy dispatch

void FluidSolver::step() {
    if (particles_.empty()) return;

    for (Particle& p : particles_) { p.ax = p.ay = 0.0f; p.dT_dt = 0.0f; }

    grid_.build(particles_);   // O(n + C) counting sort  -- always sequential

    const HeatParams hp{params_.h, params_.alpha_T, params_.h_conv, params_.mass};

    switch (params_.policy) {

    case num::Backend::omp:
        backends::omp::compute_density_pressure(particles_, params_, grid_);
        backends::omp::compute_forces(particles_, params_, grid_);
        backends::omp::heat_compute(particles_, bodies_, grid_, hp);
        backends::omp::body_collisions(particles_, bodies_, params_);
        backends::omp::integrate(particles_, params_);
        backends::omp::enforce_boundaries(particles_, params_);
        backends::omp::update_temp_range(particles_, bodies_, T_min_, T_max_);
        backends::seq::integrate_bodies(bodies_, params_);  // M bodies tiny  -- seq
        break;

    default:  // seq: Newton pair traversal
        backends::seq::compute_density_pressure(particles_, params_, grid_);
        backends::seq::compute_forces(particles_, params_, grid_);
        backends::seq::heat_compute(particles_, bodies_, grid_, hp);
        backends::seq::body_collisions(particles_, bodies_, params_);
        backends::seq::integrate(particles_, params_);
        backends::seq::enforce_boundaries(particles_, params_);
        backends::seq::update_temp_range(particles_, bodies_, T_min_, T_max_);
        backends::seq::integrate_bodies(bodies_, params_);
        break;
    }
}

} // namespace physics
