/// @file verlet_list.hpp
/// @brief Verlet neighbour list with skin-radius temporal caching
#pragma once

#include "core/types.hpp"
#include "spatial/concepts.hpp"

#include "cell_list.hpp"
#include <cmath>
#include <utility>
#include <vector>

namespace num {

template <scalars::field scalar>
/// Cached 2D neighbor lists valid until motion consumes half the skin distance.
class verlet_list_2d {
  public:
    /// Configure the physical cutoff and rebuild skin.
    verlet_list_2d(scalar cutoff, scalar skin)
        : cutoff_(cutoff), skin_(skin), ext_sq_((cutoff + skin) * (cutoff + skin)) {}

    /// @brief Build the neighbour list using a pre-built cell_list_2d.
    template <position_accessor_2d<scalar> PosAccessor>
    void build(PosAccessor &&get_pos, int n, const cell_list_2d<scalar> &cl) {
        starts_.resize(n + 1);
        flat_.clear();
        ref_x_.resize(n);
        ref_y_.resize(n);

        starts_[0] = 0;
        for (int i = 0; i < n; ++i) {
            auto [xi, yi] = get_pos(i);
            ref_x_[i] = xi;
            ref_y_[i] = yi;

            cl.query(xi, yi, [&](int j) {
                if (j == i) {
                    return;
                }
                auto [xj, yj] = get_pos(j);
                const scalar dx = xi - xj, dy = yi - yj;
                if ((dx * dx) + (dy * dy) < ext_sq_) {
                    flat_.push_back(j);
                }
            });

            starts_[i + 1] = static_cast<int>(flat_.size());
        }
    }

    /// @brief Return true if a particle moved more than half the skin.
    template <position_accessor_2d<scalar> PosAccessor>
    bool needs_rebuild(PosAccessor &&get_pos, int n) const {
        if (ref_x_.empty()) {
            return true;
        }
        const scalar half_skin_sq = (skin_ * scalar(0.5)) * (skin_ * scalar(0.5));
        for (int i = 0; i < n; ++i) {
            auto [xi, yi] = get_pos(i);
            const scalar dx = xi - ref_x_[i];
            const scalar dy = yi - ref_y_[i];
            if ((dx * dx) + (dy * dy) > half_skin_sq) {
                return true;
            }
        }
        return false;
    }

    /// @brief Cached neighbors of particle i.
    [[nodiscard]] integer_range neighbors(int i) const noexcept {
        return {flat_.data() + starts_[i], flat_.data() + starts_[i + 1]};
    }

    /// Return the physical interaction cutoff.
    scalar cutoff() const noexcept { return cutoff_; }
    /// Return the displacement buffer used between rebuilds.
    scalar skin() const noexcept { return skin_; }
    /// Return cutoff plus skin, used when constructing the cached list.
    scalar ext_cutoff() const noexcept { return cutoff_ + skin_; }
    [[nodiscard]] int n_particles() const noexcept {
        return starts_.empty() ? 0 : static_cast<int>(starts_.size()) - 1;
    }

  private:
    scalar cutoff_, skin_, ext_sq_;
    array<int> flat_;
    array<int> starts_;
    array<scalar> ref_x_;
    array<scalar> ref_y_;
};

} // namespace num
