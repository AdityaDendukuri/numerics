/// @file backends/omp/impl.hpp
/// @brief Private declarations for the OpenMP SPH backend.
/// Only included by apps/fluid_sim/fluid.cpp and heat.cpp (dispatch).
///
/// Note: integrate_bodies is absent  -- the dispatch falls through to seq
/// (identical to dispatch.cpp: `case Backend::omp: backends::seq::norm(x)`).
#pragma once

#include "particle.hpp"
#include "rigid_body.hpp"
#include "spatial_hash.hpp"
#include "fluid.hpp"
#include "heat.hpp"
#include <vector>

namespace physics::backends::omp {

/// Per-particle query  -- each thread writes only to particles[i]. O(n*k).
void compute_density_pressure(std::vector<Particle>& particles,
                               const FluidParams&     params,
                               const SpatialHash&     grid);

/// Per-particle query  -- each thread writes only to particles[i]. O(n*k).
void compute_forces(std::vector<Particle>& particles,
                    const FluidParams&     params,
                    const SpatialHash&     grid);

void body_collisions(std::vector<Particle>&        particles,
                     const std::vector<RigidBody>& bodies,
                     const FluidParams&             params);

void integrate(std::vector<Particle>& particles, const FluidParams& params);

void enforce_boundaries(std::vector<Particle>& particles, const FluidParams& params);

/// reduction(min:T_min) reduction(max:T_max)
void update_temp_range(const std::vector<Particle>&        particles,
                       const std::vector<RigidBody>&       bodies,
                       float& T_min, float& T_max);

void heat_compute(std::vector<Particle>&        particles,
                  const std::vector<RigidBody>& bodies,
                  const SpatialHash&             grid,
                  const HeatParams&              params);

} // namespace physics::backends::omp
