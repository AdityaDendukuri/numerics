/// @file verlet_list.hpp
/// @brief Verlet neighbour list with skin-radius temporal caching
#pragma once

#include "cell_list.hpp"
#include <cmath>
#include <utility>
#include <vector>

namespace num {

template<typename Scalar>
/// Cached 2D neighbor lists valid until motion consumes half the skin distance.
class VerletList2D {
public:
  /// Configure the physical cutoff and rebuild skin.
  VerletList2D(Scalar cutoff, Scalar skin)
      : cutoff_(cutoff),
        skin_(skin),
        ext_sq_((cutoff + skin) * (cutoff + skin)) {}

  /// @brief Build the neighbour list using a pre-built CellList2D.
  template<typename PosAccessor>
  void build(PosAccessor&& get_pos, int n, const CellList2D<Scalar>& cl) {
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
        const Scalar dx = xi - xj, dy = yi - yj;
        if ((dx * dx) + (dy * dy) < ext_sq_) {
          flat_.push_back(j);
        }
      });

      starts_[i + 1] = static_cast<int>(flat_.size());
    }
  }

  /// @brief Return true if a particle moved more than half the skin.
  template<typename PosAccessor>
  bool needs_rebuild(PosAccessor&& get_pos, int n) const {
    if (ref_x_.empty()) {
      return true;
    }
    const Scalar half_skin_sq = (skin_ * Scalar(0.5)) * (skin_ * Scalar(0.5));
    for (int i = 0; i < n; ++i) {
      auto [xi, yi] = get_pos(i);
      const Scalar dx = xi - ref_x_[i];
      const Scalar dy = yi - ref_y_[i];
      if ((dx * dx) + (dy * dy) > half_skin_sq) {
        return true;
      }
    }
    return false;
  }

  /// @brief Cached neighbors of particle i.
  [[nodiscard]] IntRange neighbors(int i) const noexcept {
    return {flat_.data() + starts_[i], flat_.data() + starts_[i + 1]};
  }

  /// Return the physical interaction cutoff.
  Scalar cutoff() const noexcept { return cutoff_; }
  /// Return the displacement buffer used between rebuilds.
  Scalar skin() const noexcept { return skin_; }
  /// Return cutoff plus skin, used when constructing the cached list.
  Scalar ext_cutoff() const noexcept { return cutoff_ + skin_; }
  [[nodiscard]] int n_particles() const noexcept {
    return starts_.empty() ? 0 : static_cast<int>(starts_.size()) - 1;
  }

private:
  Scalar cutoff_, skin_, ext_sq_;
  std::vector<int> flat_;
  std::vector<int> starts_;
  std::vector<Scalar> ref_x_;
  std::vector<Scalar> ref_y_;
};

} // namespace num
