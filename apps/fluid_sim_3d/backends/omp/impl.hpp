/// @file backends/omp/impl.hpp
/// @brief Private declarations for the OpenMP 3D SPH backend.
/// Only included by apps/fluid_sim_3d/fluid3d.cpp (dispatch).
///
/// integrate_bodies absent  -- dispatch falls through to seq (M bodies tiny).
#pragma once

#include "particle3d.hpp"
#include "rigid_body3d.hpp"
#include "spatial_hash3d.hpp"
#include "fluid3d.hpp"
#include "heat3d.hpp"
#include <vector>

namespace physics::backends::omp {

void compute_density_pressure(std::vector<Particle3D>&      particles,
                               const FluidParams3D&          params,
                               const SpatialHash3D&          grid);

void compute_forces(std::vector<Particle3D>&      particles,
                    const FluidParams3D&          params,
                    const SpatialHash3D&          grid);

void body_collisions(std::vector<Particle3D>&        particles,
                     const std::vector<RigidBody3D>& bodies,
                     const FluidParams3D&             params);

void integrate(std::vector<Particle3D>& particles, const FluidParams3D& params);

void enforce_boundaries(std::vector<Particle3D>& particles, const FluidParams3D& params);

void update_temp_range(const std::vector<Particle3D>&        particles,
                       const std::vector<RigidBody3D>&       bodies,
                       float& T_min, float& T_max);

void heat_compute(std::vector<Particle3D>&        particles,
                  const std::vector<RigidBody3D>& bodies,
                  const SpatialHash3D&             grid,
                  const HeatParams3D&              params);

} // namespace physics::backends::omp
