/// @file cell_list.hpp
/// @brief Cache-coherent 2D cell list for O(1) amortized neighbour queries
///
/// @par Algorithm (CS theory: counting sort + prefix sums)
///
/// The standard spatial hash maps cell -> linked list of particles.
/// Each linked-list traversal chases pointers through scattered heap memory,
/// causing a cache miss per particle in the inner neighbour loop.
///
/// The cell list replaces this with a **counting sort**:
///
///   build()  -- O(n + C)  where C = total grid cells
///     1. Assign each particle i to cell id_i = cy_i*nx + cx_i        O(n)
///     2. count[id]++ for each particle                                O(n)
///     3. Prefix sum: start[c] = sum(count[0..c-1])                       O(C)
///     4. Scatter: sorted[start[id_i] + offset++] = i                 O(n)
///     -> sorted[] contains particle indices grouped by cell.
///        start[c]..start[c+1] is the contiguous range for cell c.
///
///   query(px, py, f)  -- O(k) sequential reads
///     For each of the 9 cells in the 3x3 neighbourhood:
///       iterate sorted[start[c]..start[c+1])   -- sequential memory
///     No pointer chasing. Particles in the same cell sit next to each
///     other in sorted[], so the processor prefetches them correctly.
///
///   iterate_pairs(f)  -- O(n*k/2)  Newton's 3rd law half-shell
///     Visits each unique unordered pair {i,j} exactly once.
///     Based on the "forward half-shell" technique from molecular dynamics
///     (Plimpton 1995, LAMMPS):
///
///       for each cell (cx, cy):
///         intra-cell:  pairs (a, b) where a < b in sorted[] index
///         inter-cell:  cross-product with 4 "forward" neighbours:
///                        (cx+1, cy-1), (cx+1, cy), (cx+1, cy+1), (cx, cy+1)
///
///     These 4 offsets + self cover all 9 neighbour directions uniquely:
///       - dx > 0  ->  handled when processing the left cell
///       - dx = 0, dy > 0  ->  handled by (0,+1) offset
///       - dx < 0 or dy < 0  ->  handled from the other side
///     Every pair appears in exactly one (cell, forward-offset) combination.
///
/// @par Caller responsibilities
///   - Particles must lie within the domain passed to the constructor.
///   - get_pos(i) callable must return {x, y} for particle i.
///   - Distance check (r < cutoff) must still be done in the callback.
///
/// @tparam Scalar  float or double
#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <utility>

namespace num {

/// Lightweight read-only range over a contiguous int array (C++17-safe span).
struct IntRange {
    const int* first;
    const int* last;
    const int* begin() const noexcept { return first; }
    const int* end()   const noexcept { return last;  }
    int size()         const noexcept { return static_cast<int>(last - first); }
    bool empty()       const noexcept { return first == last; }
};

template<typename Scalar>
class CellList2D {
public:
    /// @param cell_size  Width of one cell (use kernel support radius 2h).
    /// @param xmin,xmax,ymin,ymax  Simulation domain. Particles outside
    ///        are clamped to the boundary cell (safe, just no missed neighbours).
    CellList2D(Scalar cell_size,
               Scalar xmin, Scalar xmax,
               Scalar ymin, Scalar ymax)
        : cs_(cell_size), xmin_(xmin), ymin_(ymin)
    {
        // +2 padding cells on each axis prevents boundary checks in inner loops
        nx_ = static_cast<int>(std::ceil((xmax - xmin) / cs_)) + 2;
        ny_ = static_cast<int>(std::ceil((ymax - ymin) / cs_)) + 2;
        const int total = nx_ * ny_;
        start_.assign(total + 1, 0);
        count_.assign(total, 0);
    }

    /// @brief Rebuild the cell list from n particles.
    ///
    /// PosAccessor: callable int -> std::pair<Scalar,Scalar> (x, y)
    ///
    /// Complexity: O(n + C)  (two passes over particles + one prefix sum)
    template<typename PosAccessor>
    void build(PosAccessor&& get_pos, int n) {
        sorted_.resize(n);
        const int total = nx_ * ny_;

        // Pass 1: count particles per cell
        std::fill(count_.begin(), count_.end(), 0);
        for (int i = 0; i < n; ++i)
            ++count_[cell_id_of(get_pos(i))];

        // Prefix sum -> start_[c] = first index in sorted_ for cell c
        start_[0] = 0;
        for (int c = 0; c < total; ++c)
            start_[c + 1] = start_[c] + count_[c];

        // Pass 2: scatter into sorted[] (counting sort, stable)
        std::fill(count_.begin(), count_.end(), 0);
        for (int i = 0; i < n; ++i) {
            const int cid = cell_id_of(get_pos(i));
            sorted_[start_[cid] + count_[cid]] = i;
            ++count_[cid];
        }
    }

    /// @brief Point query: calls f(int j) for every particle in the 3x3
    ///        cell neighbourhood of (px, py).
    ///
    /// Caller must still verify |r_ij| < cutoff  -- this returns candidates.
    template<typename F>
    void query(Scalar px, Scalar py, F&& f) const {
        const int cx = cell_x(px);
        const int cy = cell_y(py);
        for (int dy = -1; dy <= 1; ++dy) {
            const int qy = cy + dy;
            if (qy < 0 || qy >= ny_) continue;
            for (int dx = -1; dx <= 1; ++dx) {
                const int qx = cx + dx;
                if (qx < 0 || qx >= nx_) continue;
                const int cid = qy * nx_ + qx;
                for (int k = start_[cid]; k < start_[cid + 1]; ++k)
                    f(sorted_[k]);
            }
        }
    }

    /// @brief Newton's 3rd law pair traversal.
    ///
    /// Calls f(int i, int j) for every unique unordered pair {i,j} whose
    /// cells lie within the 3x3 neighbourhood.  Each pair appears exactly
    /// once.  Caller can apply equal-and-opposite contributions to i and j.
    ///
    /// Complexity: O(n*k/2) where k = average neighbour count.
    template<typename F>
    void iterate_pairs(F&& f) const {
        // 4 "forward" inter-cell offsets that, together with intra-cell
        // pairs, cover all unique pairs in the 3x3 neighbourhood exactly once.
        static constexpr int FDX[4] = {+1,  0, +1, -1};
        static constexpr int FDY[4] = { 0, +1, +1, +1};

        for (int cy = 0; cy < ny_; ++cy) {
            for (int cx = 0; cx < nx_; ++cx) {
                const int cid = cy * nx_ + cx;
                const int beg = start_[cid];
                const int end = start_[cid + 1];
                if (beg == end) continue;

                // 1. Intra-cell: pairs where a comes before b in sorted[]
                for (int a = beg; a < end; ++a) {
                    for (int b = a + 1; b < end; ++b) {
                        f(sorted_[a], sorted_[b]);
                    }
                }

                // 2. Inter-cell: self x each forward neighbour cell
                for (int d = 0; d < 4; ++d) {
                    const int ncx = cx + FDX[d];
                    const int ncy = cy + FDY[d];
                    if (ncx < 0 || ncx >= nx_ || ncy < 0 || ncy >= ny_) continue;
                    const int ncid = ncy * nx_ + ncx;
                    const int nbeg = start_[ncid];
                    const int nend = start_[ncid + 1];
                    if (nbeg == nend) continue;
                    for (int a = beg; a < end; ++a) {
                        for (int b = nbeg; b < nend; ++b) {
                            f(sorted_[a], sorted_[b]);
                        }
                    }
                }
            }
        }
    }

    /// @brief Direct access to sorted particle indices for cell (cx, cy).
    IntRange cell_particles(int cx, int cy) const noexcept {
        const int cid = cy * nx_ + cx;
        return { sorted_.data() + start_[cid],
                 sorted_.data() + start_[cid + 1] };
    }

    int nx() const noexcept { return nx_; }
    int ny() const noexcept { return ny_; }
    int n_particles() const noexcept { return static_cast<int>(sorted_.size()); }

private:
    Scalar cs_, xmin_, ymin_;
    int    nx_, ny_;

    std::vector<int> sorted_; ///< Particle indices sorted by cell id
    std::vector<int> start_;  ///< start_[c] = first position in sorted_ for cell c
    std::vector<int> count_;  ///< Counting-sort scratch buffer

    int cell_x(Scalar x) const noexcept {
        // +1 for padding offset; clamp to [0, nx_-1]
        const int cx = static_cast<int>(std::floor((x - xmin_) / cs_)) + 1;
        return cx < 0 ? 0 : (cx >= nx_ ? nx_ - 1 : cx);
    }
    int cell_y(Scalar y) const noexcept {
        const int cy = static_cast<int>(std::floor((y - ymin_) / cs_)) + 1;
        return cy < 0 ? 0 : (cy >= ny_ ? ny_ - 1 : cy);
    }
    int cell_id_of(std::pair<Scalar, Scalar> p) const noexcept {
        return cell_y(p.second) * nx_ + cell_x(p.first);
    }
};

} // namespace num
