/// @file spatial_hash.hpp
/// @brief SPH neighbour search  -- now powered by num::CellList2D
///
/// Replaces the original chained hash table with the counting-sort cell list
/// from src/spatial/cell_list.hpp.  The public interface is unchanged so that
/// heat.cpp keeps working without modification.
///
/// Improvements over the old chained-list hash:
///
///   Old (chained hash table)
///     build  : O(n)  -- fill table[] + next[] via random writes
///     query  : O(9 * avg_bucket)  -- pointer-chase linked list in each of 9 cells
///
///   New (counting sort cell list)
///     build  : O(n + C)  -- two sequential passes + one prefix sum
///     query  : O(k) sequential reads  -- particles in same cell are contiguous
///     pairs  : iterate_pairs()  -- visits each (i,j) pair once (Newton 3rd law)
///
/// The key cache-behaviour difference: the old next_[] array creates a random
/// walk through the particle array.  sorted_[] in CellList2D is laid out so
/// that all particles in a cell sit next to each other in memory.
#pragma once

#include "particle.hpp"
#include "spatial/cell_list.hpp"
#include <vector>

namespace physics {

class SpatialHash {
public:
    /// @param cell_size   Kernel support radius (2h).
    /// @param xmin..ymax  Simulation domain  -- required by CellList2D for the
    ///                    flat grid (no hash collisions, zero false positives).
    SpatialHash(float cell_size,
                float xmin, float xmax,
                float ymin, float ymax)
        : cl_(cell_size, xmin, xmax, ymin, ymax) {}

    /// Rebuild from the particle array.  O(n + C).
    void build(const std::vector<Particle>& particles) {
        const int n = static_cast<int>(particles.size());
        cl_.build([&](int i) {
            return std::make_pair(particles[i].x, particles[i].y);
        }, n);
    }

    /// Point query: calls f(int j) for every candidate in the 3x3 neighbourhood.
    /// Caller must verify |r_ij| < cutoff.  (Same contract as before.)
    template<typename F>
    void query(float px, float py, F&& f) const {
        cl_.query(px, py, std::forward<F>(f));
    }

    /// Newton's 3rd law pair traversal.
    /// Calls f(int i, int j) for every unique unordered pair {i,j} whose cells
    /// are within one cell of each other.  Each pair appears exactly once.
    /// Caller can apply equal-and-opposite contributions to both i and j.
    template<typename F>
    void iterate_pairs(F&& f) const {
        cl_.iterate_pairs(std::forward<F>(f));
    }

private:
    num::CellList2D<float> cl_;
};

} // namespace physics
