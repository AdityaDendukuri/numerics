/// @file fields/field3d.hpp
/// @brief 3D scalar and vector fields on uniform grids.
#pragma once

#include "core/types.hpp"
#include "container/vector_ops.hpp"

#include "container/vector.hpp"
#include "fields/grid3d.hpp"
#include <array>
#include <utility>

namespace num {

/// @brief 3D scalar field: \f$\phi_{ijk}\f$ on a uniform grid.
class scalar_field_3d {
  public:
    /// @brief Zero field, `nx x ny x nz`, spacing `dx`, origin \f$(o_x, o_y, o_z)\f$.
    scalar_field_3d(int nx, int ny, int nz, float dx, float ox = 0.0f, float oy = 0.0f,
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
    /// @brief Field sampled from `f(i, j, k)`.
    scalar_field_3d(int nx, int ny, int nz, float dx, F &&f, float ox = 0.0f, float oy = 0.0f,
                  float oz = 0.0f)
        : scalar_field_3d(nx, ny, nz, dx, ox, oy, oz) {
        fill(std::forward<F>(f));
    }

    /// @brief Grid geometry.
    [[nodiscard]] const grid_3d &grid() const { return grid_; }

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

    /// @brief Constant fill: \f$\phi_{ijk} \leftarrow v\f$.
    void fill(double v) {
        for (idx n = 0; n < data_.size(); ++n) {
            data_[n] = static_cast<real>(v);
        }
    }

    /// @brief Fill: \f$\phi_{ijk} \leftarrow f(i, j, k)\f$.
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

    /// @brief Values as a vector, in grid order.
    vec<real> &as_vec() { return data_; }
    [[nodiscard]] const vec<real> &as_vec() const { return data_; }
    real *data() { return data_.data(); }
    [[nodiscard]] const real *data() const { return data_.data(); }
    [[nodiscard]] idx size() const { return data_.size(); }

    /// @brief Trilinear interpolation: \f$\phi(x, y, z)\f$.
    [[nodiscard]] float sample(float x, float y, float z) const;

  private:
    grid_3d grid_;
    vec<real> data_;
};

/// @brief 3D vector field: three components on one grid.
struct vector_field_3d {
    scalar_field_3d x, y, z;

    vector_field_3d(int nx, int ny, int nz, float dx, float ox = 0.0f, float oy = 0.0f,
                  float oz = 0.0f);

    /// @brief Trilinear interpolation of all components at \f$(x, y, z)\f$.
    [[nodiscard]] std::array<float, 3> sample(float px, float py, float pz) const;

    /// @brief Scaling: \f$F \leftarrow s F\f$.
    void scale(float s);
};



inline float scalar_field_3d::sample(float x, float y, float z) const {
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

inline vector_field_3d::vector_field_3d(int nx, int ny, int nz, float dx, float ox, float oy, float oz)
    : x(nx, ny, nz, dx, ox, oy, oz), y(nx, ny, nz, dx, ox, oy, oz), z(nx, ny, nz, dx, ox, oy, oz) {}

inline std::array<float, 3> vector_field_3d::sample(float px, float py, float pz) const {
    return {x.sample(px, py, pz), y.sample(px, py, pz), z.sample(px, py, pz)};
}

inline void vector_field_3d::scale(float s) {
    num::scale(x.as_vec(), static_cast<real>(s));
    num::scale(y.as_vec(), static_cast<real>(s));
    num::scale(z.as_vec(), static_cast<real>(s));
}

} // namespace num
