/// @file spatial/grid3d.hpp
/// @brief 3D Cartesian scalar grid backed by num::Vector storage.
///
/// Flat layout: idx = k*ny*nx + j*nx + i  (x is fastest, z is slowest)
/// Used by the field physics module for Poisson/diffusion solves.
#pragma once

#include "core/types.hpp"
#include "core/vector.hpp"
#include <vector>
#include <utility>

namespace num {

class Grid3D {
  public:
    /// @param nx,ny,nz  Number of cells in each dimension
    /// @param dx        Uniform cell size [m]
    Grid3D(int nx, int ny, int nz, double dx = 1.0);

    int nx() const {
        return nx_;
    }
    int ny() const {
        return ny_;
    }
    int nz() const {
        return nz_;
    }
    double dx() const {
        return dx_;
    }
    int size() const {
        return nx_ * ny_ * nz_;
    }

    real& operator()(int i, int j, int k) {
        return data_[flat(i, j, k)];
    }
    real operator()(int i, int j, int k) const {
        return data_[flat(i, j, k)];
    }

    void set(int i, int j, int k, real v) {
        data_[flat(i, j, k)] = v;
    }
    void fill(real v);

    /// Fill every cell with f(i, j, k).
    template<typename F>
    void fill(F&& f) {
        for (int k = 0; k < nz_; ++k)
            for (int j = 0; j < ny_; ++j)
                for (int i = 0; i < nx_; ++i)
                    data_[flat(i, j, k)] = f(i, j, k);
    }

    /// Construct and fill from callable f(i, j, k) -> real.
    template<typename F>
    Grid3D(int nx, int ny, int nz, double dx, F&& f)
        : Grid3D(nx, ny, nz, dx) { fill(std::forward<F>(f)); }

    /// Copy contents into a new Vector (for solver interop).
    Vector to_vector() const;
    /// Copy solver result back into grid.
    void from_vector(const Vector& v);

  private:
    int               nx_, ny_, nz_;
    double            dx_;
    std::vector<real> data_; // plain array for easy indexing

    idx flat(int i, int j, int k) const {
        return (idx)(k * ny_ * nx_ + j * nx_ + i);
    }
};

} // namespace num
