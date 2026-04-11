/// @file pde/scalar_field_2d.hpp
/// @brief Scalar field on a 2D uniform interior grid.
///
/// ScalarField2D owns a Grid2D (geometry) and a flat Vector (values).
/// Node (i,j) sits at ((i+1)*h, (j+1)*h); the boundary ring is implicitly
/// zero (Dirichlet) -- BCs are enforced by the operator, not the field.
///
/// @code
///   num::Grid2D grid{64, 1.0/65};
///   num::ScalarField2D u(grid, [=](double x, double y) {
///       return num::gaussian2d(x, y, 0.5, 0.5, 0.06);
///   });
/// @endcode
#pragma once

#include "fields/grid2d.hpp"
#include "core/vector.hpp"

namespace num {

class ScalarField2D {
  public:
    explicit ScalarField2D(Grid2D g)
        : grid_(g)
        , data_(static_cast<idx>(g.size())) {}

    /// Construct and fill from callable f(x, y) -> real.
    template<typename F>
    ScalarField2D(Grid2D g, F&& f) : ScalarField2D(g) { fill(std::forward<F>(f)); }

    const Grid2D& grid() const { return grid_; }
    int    N() const { return grid_.N; }
    double h() const { return grid_.h; }

    real& operator()(int i, int j)       { return data_[static_cast<idx>(i) * grid_.N + j]; }
    real  operator()(int i, int j) const { return data_[static_cast<idx>(i) * grid_.N + j]; }

    /// Fill every interior node (i,j) with f((i+1)*h, (j+1)*h).
    template<typename F>
    void fill(F&& f) {
        for (int i = 0; i < grid_.N; ++i)
            for (int j = 0; j < grid_.N; ++j)
                data_[static_cast<idx>(i) * grid_.N + j] = f(grid_.x(i), grid_.y(j));
    }

    /// Satisfy VecField concept: exposes the underlying flat vector.
    Vector&       vec()       { return data_; }
    const Vector& vec() const { return data_; }

    real*       data()       { return data_.data(); }
    const real* data() const { return data_.data(); }
    idx         size() const { return data_.size(); }

  private:
    Grid2D grid_;
    Vector data_;
};

} // namespace num
