/// @file fields/field3d.hpp
/// @brief 3D scalar and vector fields on uniform Cartesian grids.
///
/// A ScalarField3D is geometry (Grid3D) + values (num::Vector), mirroring the
/// 2D ScalarField2D design. Because the values live in a num::Vector, the field
/// plugs straight into linear solvers and operators via .vec() with no copy.
#pragma once

#include "core/vector.hpp"
#include "fields/grid3d.hpp"
#include <array>
#include <utility>

namespace num {

class ScalarField3D {
public:
  ScalarField3D(int nx,
                int ny,
                int nz,
                float dx,
                float ox = 0.0f,
                float oy = 0.0f,
                float oz = 0.0f)
      : grid_{nx,
              ny,
              nz,
              static_cast<double>(dx),
              static_cast<double>(ox),
              static_cast<double>(oy),
              static_cast<double>(oz)},
        data_(static_cast<idx>(grid_.size())) {}

  template<typename F>
  ScalarField3D(int nx,
                int ny,
                int nz,
                float dx,
                F&& f,
                float ox = 0.0f,
                float oy = 0.0f,
                float oz = 0.0f)
      : ScalarField3D(nx, ny, nz, dx, ox, oy, oz) {
    fill(std::forward<F>(f));
  }

  const Grid3D& grid() const { return grid_; }

  int nx() const { return grid_.nx; }
  int ny() const { return grid_.ny; }
  int nz() const { return grid_.nz; }
  float dx() const { return static_cast<float>(grid_.dx); }
  float ox() const { return static_cast<float>(grid_.ox); }
  float oy() const { return static_cast<float>(grid_.oy); }
  float oz() const { return static_cast<float>(grid_.oz); }

  real& operator()(int i, int j, int k) { return data_[grid_.flat(i, j, k)]; }
  real operator()(int i, int j, int k) const { return data_[grid_.flat(i, j, k)]; }

  void set(int i, int j, int k, double v) {
    data_[grid_.flat(i, j, k)] = static_cast<real>(v);
  }

  void fill(double v) {
    for (idx n = 0; n < data_.size(); ++n)
      data_[n] = static_cast<real>(v);
  }

  /// Fill every node with f(i, j, k).
  template<typename F>
  void fill(F&& f) {
    for (int k = 0; k < grid_.nz; ++k)
      for (int j = 0; j < grid_.ny; ++j)
        for (int i = 0; i < grid_.nx; ++i)
          data_[grid_.flat(i, j, k)] = static_cast<real>(f(i, j, k));
  }

  Vector& vec() { return data_; }
  const Vector& vec() const { return data_; }
  real* data() { return data_.data(); }
  const real* data() const { return data_.data(); }
  idx size() const { return data_.size(); }

  float sample(float x, float y, float z) const;

private:
  Grid3D grid_;
  Vector data_;
};

struct VectorField3D {
  ScalarField3D x, y, z;

  VectorField3D(int nx,
                int ny,
                int nz,
                float dx,
                float ox = 0.0f,
                float oy = 0.0f,
                float oz = 0.0f);

  std::array<float, 3> sample(float px, float py, float pz) const;

  void scale(float s);
};

} // namespace num
