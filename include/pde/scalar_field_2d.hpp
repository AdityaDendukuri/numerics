/// @file pde/grid2d.hpp
/// @brief 2D uniform interior grid for PDE solvers.
///
/// ScalarField2D carries its own N and h so they don't need to be threaded through
/// every function call.  Node (i,j) sits at ((i+1)*h, (j+1)*h); the boundary
/// ring is implicitly zero (Dirichlet).
///
/// @code
///   num::ScalarField2D u(N, h);
///   num::fill_grid(u, [=](double x, double y){ return num::gaussian2d(...); });
///   num::plt::heatmap(u);
/// @endcode
#pragma once

#include "core/vector.hpp"

namespace num {

class ScalarField2D {
  public:
    ScalarField2D(int N, double h)
        : N_(N)
        , h_(h)
        , data_(static_cast<idx>(N) * N) {}

    int    N() const { return N_; }
    double h() const { return h_; }

    /// Element access at interior node (i, j).
    real& operator()(int i, int j) {
        return data_[static_cast<idx>(i) * N_ + j];
    }
    real operator()(int i, int j) const {
        return data_[static_cast<idx>(i) * N_ + j];
    }

    /// Raw access to the underlying flat vector.
    Vector&       vec()       { return data_; }
    const Vector& vec() const { return data_; }

    /// data() and size() allow ScalarField2D to be passed to any Container-templated
    /// function (e.g. num::plt::heatmap<Container>).
    real*       data()       { return data_.data(); }
    const real* data() const { return data_.data(); }
    idx         size() const { return data_.size(); }

  private:
    int    N_;
    double h_;
    Vector data_;
};

} // namespace num
