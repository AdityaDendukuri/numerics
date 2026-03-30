/// @file cell_list_3d.hpp
/// @brief Cache-coherent 3D cell list for O(1) amortized neighbour queries
///
/// Direct extension of CellList2D to three dimensions.
///
/// @par Algorithm
///
///   build()  -- O(n + C)  counting sort over C = nx*ny*nz cells
///   query()  -- O(k) sequential reads over 3x3x3 = 27 cells
///
///   iterate_pairs()  -- forward half-shell with 13 offsets
///     In 3D there are 26 neighbour directions (3^3-1).  The 13 "forward"
///     offsets are chosen so that (A,B) is visited exactly once:
///
///       forward = (dz > 0)
///              OR (dz == 0 AND dy > 0)
///              OR (dz == 0 AND dy == 0 AND dx > 0)
///
///     This partitions all 26 directed edges into 13 unique unordered pairs.
///     The 13 offsets:
///       dz=1 layer  (all 9): (-1,-1,1)(0,-1,1)(1,-1,1)
///                             (-1, 0,1)(0, 0,1)(1, 0,1)
///                             (-1, 1,1)(0, 1,1)(1, 1,1)
///       dz=0, dy=1  (3):     (-1,1,0) (0,1,0) (1,1,0)
///       dz=0, dy=0, dx=1 (1):  (1,0,0)
///
/// @tparam Scalar  float or double
#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <tuple>

namespace num {

template<typename Scalar>
class CellList3D {
public:
    /// @param cell_size  Width of one cell (use 2h  -- kernel support radius).
    CellList3D(Scalar cell_size,
               Scalar xmin, Scalar xmax,
               Scalar ymin, Scalar ymax,
               Scalar zmin, Scalar zmax)
        : cs_(cell_size)
        , xmin_(xmin), ymin_(ymin), zmin_(zmin)
    {
        nx_ = static_cast<int>(std::ceil((xmax - xmin) / cs_)) + 2;
        ny_ = static_cast<int>(std::ceil((ymax - ymin) / cs_)) + 2;
        nz_ = static_cast<int>(std::ceil((zmax - zmin) / cs_)) + 2;
        const int total = nx_ * ny_ * nz_;
        start_.assign(total + 1, 0);
        count_.assign(total, 0);
    }

    /// Build in O(n + C).
    /// PosAccessor: callable int -> std::tuple<Scalar,Scalar,Scalar>  (x,y,z)
    template<typename PosAccessor>
    void build(PosAccessor&& get_pos, int n) {
        sorted_.resize(n);
        const int total = nx_ * ny_ * nz_;

        std::fill(count_.begin(), count_.end(), 0);
        for (int i = 0; i < n; ++i)
            ++count_[cell_id_of(get_pos(i))];

        start_[0] = 0;
        for (int c = 0; c < total; ++c)
            start_[c + 1] = start_[c] + count_[c];

        std::fill(count_.begin(), count_.end(), 0);
        for (int i = 0; i < n; ++i) {
            const int cid = cell_id_of(get_pos(i));
            sorted_[start_[cid] + count_[cid]] = i;
            ++count_[cid];
        }
    }

    /// 3x3x3 neighbourhood query around (px, py, pz).
    template<typename F>
    void query(Scalar px, Scalar py, Scalar pz, F&& f) const {
        const int cx = cell_x(px);
        const int cy = cell_y(py);
        const int cz = cell_z(pz);
        for (int dz = -1; dz <= 1; ++dz) {
            const int qz = cz + dz;
            if (qz < 0 || qz >= nz_) continue;
            for (int dy = -1; dy <= 1; ++dy) {
                const int qy = cy + dy;
                if (qy < 0 || qy >= ny_) continue;
                for (int dx = -1; dx <= 1; ++dx) {
                    const int qx = cx + dx;
                    if (qx < 0 || qx >= nx_) continue;
                    const int cid = (qz * ny_ + qy) * nx_ + qx;
                    for (int k = start_[cid]; k < start_[cid + 1]; ++k)
                        f(sorted_[k]);
                }
            }
        }
    }

    /// Newton's 3rd law pair traversal  -- 13 forward offsets.
    /// Calls f(int i, int j) for each unique unordered pair exactly once.
    template<typename F>
    void iterate_pairs(F&& f) const {
        // 13 forward offsets: (dz>0) | (dz==0,dy>0) | (dz==0,dy==0,dx>0)
        static constexpr int FDX[13] = {-1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1,  1};
        static constexpr int FDY[13] = {-1,-1,-1,  0, 0, 0,  1, 1, 1,  1, 1, 1,  0};
        static constexpr int FDZ[13] = { 1, 1, 1,  1, 1, 1,  1, 1, 1,  0, 0, 0,  0};

        for (int cz = 0; cz < nz_; ++cz) {
            for (int cy = 0; cy < ny_; ++cy) {
                for (int cx = 0; cx < nx_; ++cx) {
                    const int cid = (cz * ny_ + cy) * nx_ + cx;
                    const int beg = start_[cid];
                    const int end = start_[cid + 1];
                    if (beg == end) continue;

                    // Intra-cell pairs
                    for (int a = beg; a < end; ++a)
                        for (int b = a + 1; b < end; ++b)
                            f(sorted_[a], sorted_[b]);

                    // Inter-cell: self x 13 forward neighbours
                    for (int d = 0; d < 13; ++d) {
                        const int ncx = cx + FDX[d];
                        const int ncy = cy + FDY[d];
                        const int ncz = cz + FDZ[d];
                        if (ncx < 0 || ncx >= nx_ ||
                            ncy < 0 || ncy >= ny_ ||
                            ncz < 0 || ncz >= nz_) continue;
                        const int ncid = (ncz * ny_ + ncy) * nx_ + ncx;
                        const int nbeg = start_[ncid];
                        const int nend = start_[ncid + 1];
                        if (nbeg == nend) continue;
                        for (int a = beg; a < end; ++a)
                            for (int b = nbeg; b < nend; ++b)
                                f(sorted_[a], sorted_[b]);
                    }
                }
            }
        }
    }

    int nx() const noexcept { return nx_; }
    int ny() const noexcept { return ny_; }
    int nz() const noexcept { return nz_; }
    int n_particles() const noexcept { return static_cast<int>(sorted_.size()); }

private:
    Scalar cs_, xmin_, ymin_, zmin_;
    int    nx_, ny_, nz_;

    std::vector<int> sorted_, start_, count_;

    int cell_x(Scalar x) const noexcept {
        const int cx = static_cast<int>(std::floor((x - xmin_) / cs_)) + 1;
        return cx < 0 ? 0 : (cx >= nx_ ? nx_ - 1 : cx);
    }
    int cell_y(Scalar y) const noexcept {
        const int cy = static_cast<int>(std::floor((y - ymin_) / cs_)) + 1;
        return cy < 0 ? 0 : (cy >= ny_ ? ny_ - 1 : cy);
    }
    int cell_z(Scalar z) const noexcept {
        const int cz = static_cast<int>(std::floor((z - zmin_) / cs_)) + 1;
        return cz < 0 ? 0 : (cz >= nz_ ? nz_ - 1 : cz);
    }
    int cell_id_of(std::tuple<Scalar, Scalar, Scalar> p) const noexcept {
        const auto [x, y, z] = p;
        return (cell_z(z) * ny_ + cell_y(y)) * nx_ + cell_x(x);
    }
};

} // namespace num
