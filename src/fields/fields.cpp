/// @file src/fields/fields.cpp
/// @brief Out-of-line implementations for ScalarField3D and VectorField3D.
#include "fields/field3d.hpp"

#include "core/vector.hpp"

namespace num {

float ScalarField3D::sample(float x, float y, float z) const {
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
  return ((1 - tz)
          * (((1 - ty) * (((1 - tx) * v(0, 0, 0)) + (tx * v(1, 0, 0))))
             + (ty * (((1 - tx) * v(0, 1, 0)) + (tx * v(1, 1, 0))))))
         + (tz
            * (((1 - ty) * (((1 - tx) * v(0, 0, 1)) + (tx * v(1, 0, 1))))
               + (ty * (((1 - tx) * v(0, 1, 1)) + (tx * v(1, 1, 1))))));
}

VectorField3D::VectorField3D(int nx,
                             int ny,
                             int nz,
                             float dx,
                             float ox,
                             float oy,
                             float oz)
    : x(nx, ny, nz, dx, ox, oy, oz),
      y(nx, ny, nz, dx, ox, oy, oz),
      z(nx, ny, nz, dx, ox, oy, oz) {
}

std::array<float, 3> VectorField3D::sample(float px, float py, float pz) const {
  return {x.sample(px, py, pz), y.sample(px, py, pz), z.sample(px, py, pz)};
}

void VectorField3D::scale(float s) {
  num::scale(x.vec(), static_cast<real>(s));
  num::scale(y.vec(), static_cast<real>(s));
  num::scale(z.vec(), static_cast<real>(s));
}

} // namespace num
