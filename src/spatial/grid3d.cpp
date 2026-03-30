#include "spatial/grid3d.hpp"
#include <algorithm>

namespace num {

Grid3D::Grid3D(int nx, int ny, int nz, double dx)
    : nx_(nx), ny_(ny), nz_(nz), dx_(dx), data_(nx * ny * nz, 0.0) {}

void Grid3D::fill(real v) {
    std::fill(data_.begin(), data_.end(), v);
}

Vector Grid3D::to_vector() const {
    Vector v(size());
    for (int i = 0; i < size(); ++i) v[i] = data_[i];
    return v;
}

void Grid3D::from_vector(const Vector& v) {
    for (int i = 0; i < size(); ++i) data_[i] = v[i];
}

} // namespace num
