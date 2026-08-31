/// @file fields/scalar_field_2d.hpp
/// @brief Scalar field on a 2D uniform interior grid.
#pragma once

#include "container/vector.hpp"
#include "fields/grid2d.hpp"

namespace num {

/// Scalar values stored at the interior nodes of a uniform square grid.
class ScalarField2D {
  public:
    /// Allocate a zero-filled field on g.
    explicit ScalarField2D(Grid2D g) : grid_(g), data_(static_cast<idx>(g.size())) {}

    template <typename F>
    /// Sample f(x,y) at every grid node during construction.
    ScalarField2D(Grid2D g, F &&f) : ScalarField2D(g) {
        fill(std::forward<F>(f));
    }

    /// Return the field geometry.
    [[nodiscard]] const Grid2D &grid() const { return grid_; }
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
    Vector &vec() { return data_; }
    [[nodiscard]] const Vector &vec() const { return data_; }

    real *data() { return data_.data(); }
    [[nodiscard]] const real *data() const { return data_.data(); }
    [[nodiscard]] idx size() const { return data_.size(); }

  private:
    Grid2D grid_;
    Vector data_;
};

} // namespace num
