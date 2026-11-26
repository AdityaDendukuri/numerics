/// @file spatial/verlet_list.hpp
/// @brief Verlet neighbour lists with a skin radius.
#pragma once

#include "core/types.hpp"
#include "spatial/concepts.hpp"

#include "cell_list.hpp"
#include <cmath>
#include <utility>
#include <vector>

namespace num {

template <field scalar>
/// @brief List of each particle's neighbours within \f$r_c + r_s\f$, in 2D (Verlet list).
/// Valid until a particle moves \f$r_s / 2\f$.
class verlet_list_2d {
  public:
    /// @brief Construct an empty list with cutoff \f$r_c\f$ and skin \f$r_s\f$.
    verlet_list_2d(scalar cutoff, scalar skin)
        : cutoff_(cutoff), skin_(skin), ext_sq_((cutoff + skin) * (cutoff + skin)) {}

    /// @brief Build the list again from a cell list.
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

    /// @brief Test whether a particle has moved more than \f$r_s / 2\f$ since the last build.
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

    /// @brief Return the neighbours of particle `i`.
    [[nodiscard]] integer_range neighbors(int i) const noexcept {
        return {flat_.data() + starts_[i], flat_.data() + starts_[i + 1]};
    }

    /// @brief Return the cutoff \f$r_c\f$.
    scalar cutoff() const noexcept { return cutoff_; }
    /// @brief Return the skin \f$r_s\f$.
    scalar skin() const noexcept { return skin_; }
    /// @brief Return the extended cutoff \f$r_c + r_s\f$.
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
