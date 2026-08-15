/// @file fields/scalar_field_2d.hpp
/// @brief Scalar field on a 2D uniform interior grid.
#pragma once

#include "core/vector.hpp"
#include "fields/grid2d.hpp"

namespace num {

class ScalarField2D {
public:
  explicit ScalarField2D(Grid2D g)
      : grid_(g),
        data_(static_cast<idx>(g.size())) {}

  template<typename F>
  ScalarField2D(Grid2D g, F&& f)
      : ScalarField2D(g) {
    fill(std::forward<F>(f));
  }

  const Grid2D& grid() const { return grid_; }
  int N() const { return grid_.N; }
  double h() const { return grid_.h; }

  real& operator()(int i, int j) { return data_[static_cast<idx>(i) * grid_.N + j]; }
  real operator()(int i, int j) const { return data_[static_cast<idx>(i) * grid_.N + j]; }

  template<typename F>
  void fill(F&& f) {
    for (int i = 0; i < grid_.N; ++i)
      for (int j = 0; j < grid_.N; ++j)
        data_[static_cast<idx>(i) * grid_.N + j] = f(grid_.x(i), grid_.y(j));
  }

  Vector& vec() { return data_; }
  const Vector& vec() const { return data_; }

  real* data() { return data_.data(); }
  const real* data() const { return data_.data(); }
  idx size() const { return data_.size(); }

private:
  Grid2D grid_;
  Vector data_;
};

} // namespace num
