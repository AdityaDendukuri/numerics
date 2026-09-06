/// @file cell_list_3d.hpp
/// @brief Cache-coherent 3D cell list for O(1) amortized neighbour queries
#pragma once

#include "spatial/concepts.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <tuple>
#include <vector>

namespace num {

template <scalars::field scalar>
/// Counting-sorted 3D spatial bins for local-neighbor iteration.
class cell_list_3d {
  public:
    /// Cover the bounding box with padded cubic cells.
    cell_list_3d(scalar cell_size, scalar xmin, scalar xmax, scalar ymin, scalar ymax, scalar zmin,
               scalar zmax)
        : cs_(cell_size), xmin_(xmin), ymin_(ymin), zmin_(zmin),
          nx_(static_cast<int>(std::ceil((xmax - xmin) / cs_)) + 2),
          ny_(static_cast<int>(std::ceil((ymax - ymin) / cs_)) + 2),
          nz_(static_cast<int>(std::ceil((zmax - zmin) / cs_)) + 2) {

        const int total = nx_ * ny_ * nz_;
        start_.assign(total + 1, 0);
        count_.assign(total, 0);
    }

    /// @brief Rebuild by counting-sort over cell ids.
    template <position_accessor_2d<scalar> PosAccessor>
    void build(PosAccessor &&get_pos, int n) {
        sorted_.resize(n);
        const int total = nx_ * ny_ * nz_;

        std::fill(count_.begin(), count_.end(), 0);
        for (int i = 0; i < n; ++i) {
            ++count_[cell_id_of(get_pos(i))];
        }

        start_[0] = 0;
        for (int c = 0; c < total; ++c) {
            start_[c + 1] = start_[c] + count_[c];
        }

        std::fill(count_.begin(), count_.end(), 0);
        for (int i = 0; i < n; ++i) {
            const int cid = cell_id_of(get_pos(i));
            sorted_[start_[cid] + count_[cid]] = i;
            ++count_[cid];
        }
    }

    /// @brief Call f(j) for candidate particles near (px, py, pz).
    template <typename F>
    void query(scalar px, scalar py, scalar pz, F &&f) const {
        const int cx = cell_x(px);
        const int cy = cell_y(py);
        const int cz = cell_z(pz);
        for (int dz = -1; dz <= 1; ++dz) {
            const int qz = cz + dz;
            if (qz < 0 || qz >= nz_) {
                continue;
            }
            for (int dy = -1; dy <= 1; ++dy) {
                const int qy = cy + dy;
                if (qy < 0 || qy >= ny_) {
                    continue;
                }
                for (int dx = -1; dx <= 1; ++dx) {
                    const int qx = cx + dx;
                    if (qx < 0 || qx >= nx_) {
                        continue;
                    }
                    const int cid = (((qz * ny_) + qy) * nx_) + qx;
                    for (int k = start_[cid]; k < start_[cid + 1]; ++k) {
                        f(sorted_[k]);
                    }
                }
            }
        }
    }

    /// @brief Visit each candidate pair once.
    template <typename F>
    void iterate_pairs(F &&f) const {
        // Half-shell offsets cover all neighboring cell pairs once.
        static constexpr int FDX[13] = {-1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, 1};
        static constexpr int FDY[13] = {-1, -1, -1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0};
        static constexpr int FDZ[13] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0};

        for (int cz = 0; cz < nz_; ++cz) {
            for (int cy = 0; cy < ny_; ++cy) {
                for (int cx = 0; cx < nx_; ++cx) {
                    const int cid = (((cz * ny_) + cy) * nx_) + cx;
                    const int beg = start_[cid];
                    const int end = start_[cid + 1];
                    if (beg == end) {
                        continue;
                    }

                    // Intra-cell pairs
                    for (int a = beg; a < end; ++a) {
                        for (int b = a + 1; b < end; ++b) {
                            f(sorted_[a], sorted_[b]);
                        }
                    }

                    // Inter-cell: self x 13 forward neighbours
                    for (int d = 0; d < 13; ++d) {
                        const int ncx = cx + FDX[d];
                        const int ncy = cy + FDY[d];
                        const int ncz = cz + FDZ[d];
                        if (ncx < 0 || ncx >= nx_ || ncy < 0 || ncy >= ny_ || ncz < 0 ||
                            ncz >= nz_) {
                            continue;
                        }
                        const int ncid = (((ncz * ny_) + ncy) * nx_) + ncx;
                        const int nbeg = start_[ncid];
                        const int nend = start_[ncid + 1];
                        if (nbeg == nend) {
                            continue;
                        }
                        for (int a = beg; a < end; ++a) {
                            for (int b = nbeg; b < nend; ++b) {
                                f(sorted_[a], sorted_[b]);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Return cell-grid dimensions and stored particle count.
    [[nodiscard]] int nx() const noexcept { return nx_; }
    [[nodiscard]] int ny() const noexcept { return ny_; }
    [[nodiscard]] int nz() const noexcept { return nz_; }
    [[nodiscard]] int n_particles() const noexcept { return static_cast<int>(sorted_.size()); }

  private:
    scalar cs_ = 0, xmin_ = 0, ymin_ = 0, zmin_ = 0;
    int nx_ = 0, ny_ = 0, nz_ = 0;

    std::vector<int> sorted_, start_, count_;

    int cell_x(scalar x) const noexcept {
        const int cx = static_cast<int>(std::floor((x - xmin_) / cs_)) + 1;
        return cx < 0 ? 0 : (cx >= nx_ ? nx_ - 1 : cx);
    }
    int cell_y(scalar y) const noexcept {
        const int cy = static_cast<int>(std::floor((y - ymin_) / cs_)) + 1;
        return cy < 0 ? 0 : (cy >= ny_ ? ny_ - 1 : cy);
    }
    int cell_z(scalar z) const noexcept {
        const int cz = static_cast<int>(std::floor((z - zmin_) / cs_)) + 1;
        return cz < 0 ? 0 : (cz >= nz_ ? nz_ - 1 : cz);
    }
    int cell_id_of(std::tuple<scalar, scalar, scalar> p) const noexcept {
        const auto [x, y, z] = p;
        return (((cell_z(z) * ny_) + cell_y(y)) * nx_) + cell_x(x);
    }
};

} // namespace num
