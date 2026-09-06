/// @file cell_list.hpp
/// @brief Cache-coherent 2D cell list for O(1) amortized neighbour queries
#pragma once

#include "core/types.hpp"
#include "spatial/concepts.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <utility>
#include <vector>

namespace num {

/// Lightweight non-owning range of integer particle indices.
struct integer_range {
    const int *first;
    const int *last;
    [[nodiscard]] const int *begin() const noexcept { return first; }
    [[nodiscard]] const int *end() const noexcept { return last; }
    [[nodiscard]] int size() const noexcept { return static_cast<int>(last - first); }
    [[nodiscard]] bool empty() const noexcept { return first == last; }
};

template <scalars::field scalar>
/// Counting-sorted 2D spatial bins for local-neighbor iteration.
class cell_list_2d {
  public:
    /// Cover the bounding box with padded square cells.
    cell_list_2d(scalar cell_size, scalar xmin, scalar xmax, scalar ymin, scalar ymax)
        : cs_(cell_size), xmin_(xmin), ymin_(ymin) {
        // Padding cells avoid boundary checks in the hot query loops.
        nx_ = static_cast<int>(std::ceil((xmax - xmin) / cs_)) + 2;
        ny_ = static_cast<int>(std::ceil((ymax - ymin) / cs_)) + 2;
        const int total = nx_ * ny_;
        start_.assign(total + 1, 0);
        count_.assign(total, 0);
    }

    /// @brief Rebuild by counting-sort over cell ids.
    template <position_accessor_2d<scalar> PosAccessor>
    void build(PosAccessor &&get_pos, int n) {
        sorted_.resize(n);
        const int total = nx_ * ny_;

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

    /// @brief Call f(j) for candidate particles near (px, py).
    template <typename F>
    void query(scalar px, scalar py, F &&f) const {
        const int cx = cell_x(px);
        const int cy = cell_y(py);
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
                const int cid = (qy * nx_) + qx;
                for (int k = start_[cid]; k < start_[cid + 1]; ++k) {
                    f(sorted_[k]);
                }
            }
        }
    }

    /// @brief Visit each candidate pair once.
    template <typename F>
    void iterate_pairs(F &&f) const {
        // Half-shell offsets cover all neighboring cell pairs once.
        static constexpr int FDX[4] = {+1, 0, +1, -1};
        static constexpr int FDY[4] = {0, +1, +1, +1};

        for (int cy = 0; cy < ny_; ++cy) {
            for (int cx = 0; cx < nx_; ++cx) {
                const int cid = (cy * nx_) + cx;
                const int beg = start_[cid];
                const int end = start_[cid + 1];
                if (beg == end) {
                    continue;
                }

                for (int a = beg; a < end; ++a) {
                    for (int b = a + 1; b < end; ++b) {
                        f(sorted_[a], sorted_[b]);
                    }
                }

                for (int d = 0; d < 4; ++d) {
                    const int ncx = cx + FDX[d];
                    const int ncy = cy + FDY[d];
                    if (ncx < 0 || ncx >= nx_ || ncy < 0 || ncy >= ny_) {
                        continue;
                    }
                    const int ncid = (ncy * nx_) + ncx;
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

    /// Return the particles stored in one cell.
    [[nodiscard]] integer_range cell_particles(int cx, int cy) const noexcept {
        const int cid = (cy * nx_) + cx;
        return {sorted_.data() + start_[cid], sorted_.data() + start_[cid + 1]};
    }

    [[nodiscard]] int nx() const noexcept { return nx_; }
    [[nodiscard]] int ny() const noexcept { return ny_; }
    [[nodiscard]] int n_particles() const noexcept { return static_cast<int>(sorted_.size()); }

  private:
    scalar cs_ = 0, xmin_ = 0, ymin_ = 0;
    int nx_ = 0, ny_ = 0;

    array<int> sorted_;
    array<int> start_;
    array<int> count_;

    int cell_x(scalar x) const noexcept {
        const int cx = static_cast<int>(std::floor((x - xmin_) / cs_)) + 1;
        return cx < 0 ? 0 : (cx >= nx_ ? nx_ - 1 : cx);
    }
    int cell_y(scalar y) const noexcept {
        const int cy = static_cast<int>(std::floor((y - ymin_) / cs_)) + 1;
        return cy < 0 ? 0 : (cy >= ny_ ? ny_ - 1 : cy);
    }
    int cell_id_of(std::pair<scalar, scalar> p) const noexcept {
        return (cell_y(p.second) * nx_) + cell_x(p.first);
    }
};

} // namespace num
