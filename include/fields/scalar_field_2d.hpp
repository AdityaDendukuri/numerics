/// @file fields/scalar_field_2d.hpp
/// @brief scalar field on a 2D uniform interior grid.
#pragma once

#include "container/vector.hpp"
#include "fields/grid2d.hpp"

namespace num {

/// scalar values stored at the interior nodes of a uniform square grid.
class scalar_field_2d {
  public:
    /// Allocate a zero-filled field on g.
    explicit scalar_field_2d(grid2d g) : grid_(g), data_(static_cast<idx>(g.size())) {}

    template <typename F>
    /// Sample f(x,y) at every grid node during construction.
    scalar_field_2d(grid2d g, F &&f) : scalar_field_2d(g) {
        fill(std::forward<F>(f));
    }

    /// Return the field geometry.
    [[nodiscard]] const grid2d &grid() const { return grid_; }
    [[nodiscard]] int N() const { return grid_.N; }
    [[nodiscard]] double h() const { return grid_.h; }

    real &operator()(int i, int j) { return data_[(static_cast<idx>(i) * grid_.N) + j]; }
    real operator()(int i, int j) const { return data_[(static_cast<idx>(i) * grid_.N) + j]; }

    template <typename F>
    /// Replace every field value with f(x,y) at its node.
    void fill(F &&f) {
        for (int i = 0; i < grid_.N; ++i) {
            for (int j = 0; j < grid_.N; ++j) {
                data_[(static_cast<idx>(i) * grid_.N) + j] = f(grid_.x(i), grid_.y(j));
            }
        }
    }

    /// Access the contiguous values in row-major grid order.
    vec &as_vec() { return data_; }
    [[nodiscard]] const vec &as_vec() const { return data_; }

    real *data() { return data_.data(); }
    [[nodiscard]] const real *data() const { return data_.data(); }
    [[nodiscard]] idx size() const { return data_.size(); }

  private:
    grid2d grid_;
    vec data_;
};

} // namespace num
