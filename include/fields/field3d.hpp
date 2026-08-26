/// @file fields/field3d.hpp
/// @brief 3D scalar and vector fields on uniform Cartesian grids.
///
/// A ScalarField3D is geometry (Grid3D) + values (num::Vector), mirroring the
/// 2D ScalarField2D design. Because the values live in a num::Vector, the field
/// plugs straight into linear solvers and operators via .vec() with no copy.
#pragma once

#include "container/vector_ops.hpp"

#include "container/vector.hpp"
#include "fields/grid3d.hpp"
#include <array>
#include <utility>

namespace num {

/// Scalar values stored on a uniform Cartesian grid.
class ScalarField3D {
  public:
    /// Allocate a zero-filled field with optional physical origin.
    ScalarField3D(int nx, int ny, int nz, float dx, float ox = 0.0f, float oy = 0.0f,
                  float oz = 0.0f)
        : grid_{nx,
                ny,
                nz,
                static_cast<double>(dx),
                static_cast<double>(ox),
                static_cast<double>(oy),
                static_cast<double>(oz)},
          data_(static_cast<idx>(grid_.size())) {}

    template <typename F>
    /// Sample f(i,j,k) at every grid node during construction.
    ScalarField3D(int nx, int ny, int nz, float dx, F &&f, float ox = 0.0f, float oy = 0.0f,
                  float oz = 0.0f)
        : ScalarField3D(nx, ny, nz, dx, ox, oy, oz) {
        fill(std::forward<F>(f));
    }

    /// Return the field geometry.
    [[nodiscard]] const Grid3D &grid() const { return grid_; }

    [[nodiscard]] int nx() const { return grid_.nx; }
    [[nodiscard]] int ny() const { return grid_.ny; }
    [[nodiscard]] int nz() const { return grid_.nz; }
    [[nodiscard]] float dx() const { return static_cast<float>(grid_.dx); }
    [[nodiscard]] float ox() const { return static_cast<float>(grid_.ox); }
    [[nodiscard]] float oy() const { return static_cast<float>(grid_.oy); }
    [[nodiscard]] float oz() const { return static_cast<float>(grid_.oz); }

    real &operator()(int i, int j, int k) { return data_[grid_.flat(i, j, k)]; }
    real operator()(int i, int j, int k) const { return data_[grid_.flat(i, j, k)]; }

    void set(int i, int j, int k, double v) { data_[grid_.flat(i, j, k)] = static_cast<real>(v); }

    /// Fill every node with a constant value.
    void fill(double v) {
        for (idx n = 0; n < data_.size(); ++n) {
            data_[n] = static_cast<real>(v);
        }
    }

    /// Fill every node with f(i, j, k).
    template <typename F>
    void fill(F &&f) {
        for (int k = 0; k < grid_.nz; ++k) {
            for (int j = 0; j < grid_.ny; ++j) {
                for (int i = 0; i < grid_.nx; ++i) {
                    data_[grid_.flat(i, j, k)] = static_cast<real>(f(i, j, k));
                }
            }
        }
    }

    /// Access the contiguous values in grid flattening order.
    Vector &vec() { return data_; }
    [[nodiscard]] const Vector &vec() const { return data_; }
    real *data() { return data_.data(); }
    [[nodiscard]] const real *data() const { return data_.data(); }
    [[nodiscard]] idx size() const { return data_.size(); }

    /// Trilinearly interpolate the field at physical coordinates.
    [[nodiscard]] float sample(float x, float y, float z) const;

  private:
    Grid3D grid_;
    Vector data_;
};

/// Three-component vector field sharing a common 3D grid.
struct VectorField3D {
    ScalarField3D x, y, z;

    VectorField3D(int nx, int ny, int nz, float dx, float ox = 0.0f, float oy = 0.0f,
                  float oz = 0.0f);

    /// Trilinearly interpolate all components at a physical point.
    [[nodiscard]] std::array<float, 3> sample(float px, float py, float pz) const;

    /// Scale all components in place.
    void scale(float s);
};



inline float ScalarField3D::sample(float x, float y, float z) const {
    const float gx = (x - ox()) / dx();
    const float gy = (y - oy()) / dx();
    const float gz = (z - oz()) / dx();

    if (gx < 0 || gx >= nx() - 1 || gy < 0 || gy >= ny() - 1 || gz < 0 || gz >= nz() - 1) {
        return 0.0f;
    }

    const int i0 = static_cast<int>(gx);
    const int j0 = static_cast<int>(gy);
    const int k0 = static_cast<int>(gz);
    const float tx = gx - i0, ty = gy - j0, tz = gz - k0;

    auto v = [&](int di, int dj, int dk) {
        return static_cast<float>((*this)(i0 + di, j0 + dj, k0 + dk));
    };
    return ((1 - tz) * (((1 - ty) * (((1 - tx) * v(0, 0, 0)) + (tx * v(1, 0, 0)))) +
                        (ty * (((1 - tx) * v(0, 1, 0)) + (tx * v(1, 1, 0)))))) +
           (tz * (((1 - ty) * (((1 - tx) * v(0, 0, 1)) + (tx * v(1, 0, 1)))) +
                  (ty * (((1 - tx) * v(0, 1, 1)) + (tx * v(1, 1, 1))))));
}

inline VectorField3D::VectorField3D(int nx, int ny, int nz, float dx, float ox, float oy, float oz)
    : x(nx, ny, nz, dx, ox, oy, oz), y(nx, ny, nz, dx, ox, oy, oz), z(nx, ny, nz, dx, ox, oy, oz) {}

inline std::array<float, 3> VectorField3D::sample(float px, float py, float pz) const {
    return {x.sample(px, py, pz), y.sample(px, py, pz), z.sample(px, py, pz)};
}

inline void VectorField3D::scale(float s) {
    num::scale(x.vec(), static_cast<real>(s));
    num::scale(y.vec(), static_cast<real>(s));
    num::scale(z.vec(), static_cast<real>(s));
}

} // namespace num
