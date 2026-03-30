/// @file spatial_hash3d.hpp
/// @brief 3D SPH neighbour search  -- powered by num::CellList3D
///
/// Replaces the original chained hash table with the counting-sort 3D cell
/// list from include/spatial/cell_list_3d.hpp.  Public interface is unchanged
/// so all backend call sites keep working.
#pragma once

#include "particle3d.hpp"
#include "spatial/cell_list_3d.hpp"
#include <vector>

namespace physics {

class SpatialHash3D {
public:
    SpatialHash3D(float cell_size,
                  float xmin, float xmax,
                  float ymin, float ymax,
                  float zmin, float zmax)
        : cl_(cell_size, xmin, xmax, ymin, ymax, zmin, zmax) {}

    void build(const std::vector<Particle3D>& particles) {
        const int n = static_cast<int>(particles.size());
        cl_.build([&](int i) {
            return std::make_tuple(particles[i].x, particles[i].y, particles[i].z);
        }, n);
    }

    template<typename F>
    void query(float px, float py, float pz, F&& f) const {
        cl_.query(px, py, pz, std::forward<F>(f));
    }

    /// Newton's 3rd law pair traversal  -- 13 forward offsets, O(n*k/2).
    template<typename F>
    void iterate_pairs(F&& f) const {
        cl_.iterate_pairs(std::forward<F>(f));
    }

private:
    num::CellList3D<float> cl_;
};

} // namespace physics
