/// @file backends/seq/impl.hpp
/// @brief Private declarations for the sequential SPH backend.
/// Only included by apps/fluid_sim/fluid.cpp and heat.cpp (dispatch).
#pragma once

#include "particle.hpp"
#include "rigid_body.hpp"
#include "spatial_hash.hpp"
#include "fluid.hpp"
#include "heat.hpp"
#include <vector>

namespace physics::backends::seq {

/// Newton's 3rd law pair traversal  -- O(n*k/2).
void compute_density_pressure(std::vector<Particle>& particles,
                               const FluidParams&     params,
                               const SpatialHash&     grid);

/// Newton's 3rd law pair traversal  -- O(n*k/2).
void compute_forces(std::vector<Particle>& particles,
                    const FluidParams&     params,
                    const SpatialHash&     grid);

void body_collisions(std::vector<Particle>&        particles,
                     const std::vector<RigidBody>& bodies,
                     const FluidParams&             params);

void integrate(std::vector<Particle>& particles, const FluidParams& params);

void enforce_boundaries(std::vector<Particle>& particles, const FluidParams& params);

void update_temp_range(const std::vector<Particle>&        particles,
                       const std::vector<RigidBody>&       bodies,
                       float& T_min, float& T_max);

void integrate_bodies(std::vector<RigidBody>& bodies, const FluidParams& params);

void heat_compute(std::vector<Particle>&        particles,
                  const std::vector<RigidBody>& bodies,
                  const SpatialHash&             grid,
                  const HeatParams&              params);

} // namespace physics::backends::seq
